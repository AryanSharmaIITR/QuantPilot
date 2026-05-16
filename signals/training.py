import math
from sklearn.metrics import roc_auc_score,accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


from models import TimeSeriesTransformer, MarketAggregator, PositionalEncoding, FusionV2, SurrogateHead
from loadData import get_dataloader
from data import PREPROCESSED_DIR_NSE, PREPROCESSED_DIR_STOCK,DIM,NSE_INDICES,STOCK_INDICES

train_loader, val_loader, test_loader = get_dataloader(PREPROCESSED_DIR_NSE, PREPROCESSED_DIR_STOCK, batch_size=32, sequence_length=14)

# input_dim = 18
# num_indices = 11
# stock_indices = 29
class MarketPipeline:
    def __init__(
        self,
        dim: int = DIM,
        lr: float = 1e-3,
        name: str = "market_pipeline",
        nn_epochs: int = 100,
        transformer_epochs: int = 300,
        transformer_weight_decay: float = 1e-5,
        transformer_lr: float = 1e-4,
        device: str | None = None,
        xgb_params: dict | None = None,
    ):
        self.dim = dim
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lr = lr
        self.name = name
        self.transformer_weight_decay = transformer_weight_decay
        self.transformer_lr = transformer_lr
        self.transformer_epochs = transformer_epochs
        self.nn_epochs = nn_epochs

        # Build NN modules
        self.aggregator = MarketAggregator(dim).to(self.device)
        self.fusion = FusionV2(dim).to(self.device)
        self.head = SurrogateHead(270).to(self.device)

        # XGBoost params
        self.xgb_params = xgb_params or {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "seed": 42,
        }
        self.xgb_model = None

        # Create transformer models
        self.market_model = TimeSeriesTransformer(
            input_dim=dim,
            d_model=128,
            nhead=8,
            num_layers=3,
            num_market_indices=NSE_INDICES
        ).to(self.device)

        self.stock_model = TimeSeriesTransformer(
            input_dim=dim,
            d_model=128,
            nhead=8,
            num_layers=3,
            num_market_indices=STOCK_INDICES
        ).to(self.device)

    def _nn_params(self):
        return (
            list(self.aggregator.parameters())
            + list(self.fusion.parameters())
            + list(self.head.parameters())
            + list(self.market_model.parameters())  # Add these
            + list(self.stock_model.parameters())   # Add these
        )

    @torch.no_grad()
    def _extract_features(self, skd_matrix: torch.Tensor, mkd_matrix: torch.Tensor, stock_data: torch.Tensor) -> np.ndarray:
        """
        Forward pass through MarketAggregator + FusionV2 for ALL 29 stocks.

        Args:
            skd_matrix : (B, 29, 18) - Transformer outputs
            mkd_matrix : (B, 11, 18) - Market transformer outputs
            stock_data : (B, 29, 14, 18) - Raw stock data with timesteps

        Returns:
            features : (B * 29, 270) numpy array ← XGBoost input
        """
        B = skd_matrix.shape[0]
        
        # Ensure tensors are on correct device
        skd_matrix = skd_matrix.to(self.device)
        mkd_matrix = mkd_matrix.to(self.device)
        stock_data = stock_data.to(self.device)
        
        # Flatten market output: (B, 11, 18) -> (B, 198)
        flt_mkd_out = mkd_matrix.view(B, -1)  # (B, 198)
        
        # Aggregate market: (B, 11, 18) -> (B, 18)
        market_vec = self.aggregator(mkd_matrix)  # (B, 18)
        
        fused_list = []
        for stock_idx in range(STOCK_INDICES):
            s_vec = skd_matrix[:, stock_idx, :]           # (B, 18) - Transformer output
            ls_vec = stock_data[:, stock_idx, -1, :]      # (B, 18) - Last raw timestep
            fused = self.fusion(s_vec, market_vec)        # (B, 18) - Fused features
            
            # Concatenate all: 18+18+18+18+198 = 270
            input_vec = torch.cat([ls_vec, s_vec, market_vec, fused, flt_mkd_out], dim=1)
            # Shape: (B, 270)
            
            fused_list.append(input_vec)
        
        # Stack: (B, 29, 270)
        features = torch.stack(fused_list, dim=1)
        
        # Return flattened features: (B * 29, 270)
        return features.cpu().numpy().reshape(B * STOCK_INDICES, -1)  # -1 automatically gets 270
    
    # ── Phase 1: TimeSeriesTransformer pre-training ──────────────
    def pretrain_transformers(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # Use smaller learning rate
        optimizer_market = torch.optim.Adam(self.market_model.parameters(), lr=self.transformer_lr, weight_decay=self.transformer_weight_decay)
        optimizer_stock = torch.optim.Adam(self.stock_model.parameters(), lr=self.transformer_lr, weight_decay=self.transformer_weight_decay)

        criterion = nn.MSELoss()

                
        scheduler_market = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_market, mode='min', factor=0.5, patience=5)
        scheduler_stock = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_stock, mode='min', factor=0.5, patience=5)

        best_market_loss = float('inf')
        best_stock_loss = float('inf')

        train_market_losses = []
        train_stock_losses = []
        val_market_losses = []
        val_stock_losses = []

        for epoch in range(self.transformer_epochs):
            # Training phase
            self.market_model.train()
            self.stock_model.train()
            
            train_market_loss = 0
            train_stock_loss = 0
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.transformer_epochs} [Train]')
            
            for market_data, stock_data, market_target, stock_target, target in train_bar:
                # Move to device
                market_data = market_data.to(self.device)  # (batch, 11, 14, 18)
                stock_data = stock_data.to(self.device)    # (batch, 29, 14, 19)
                market_target = market_target.to(self.device)  # (batch, 11, 18)
                stock_target = stock_target.to(self.device)    # (batch, 29, 19)

                # Train Market Model
                optimizer_market.zero_grad()
                market_output = self.market_model(market_data)
                market_loss = criterion(market_output, market_target)
                market_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.market_model.parameters(), max_norm=1.0)
                optimizer_market.step()
                
                # Train Stock Model
                optimizer_stock.zero_grad()
                stock_output = self.stock_model(stock_data)
                stock_loss = criterion(stock_output, stock_target)
                stock_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.stock_model.parameters(), max_norm=1.0)
                optimizer_stock.step()
                
                train_market_loss += market_loss.item()
                train_stock_loss += stock_loss.item()
                
                train_bar.set_postfix({
                    'mkt_loss': market_loss.item(),
                    'stk_loss': stock_loss.item()
                })
            
            avg_train_market = train_market_loss / len(train_loader)
            avg_train_stock = train_stock_loss / len(train_loader)
            train_market_losses.append(avg_train_market)
            train_stock_losses.append(avg_train_stock)
            
            # Validation phase
            self.market_model.eval()
            self.stock_model.eval()
            
            val_market_loss = 0
            val_stock_loss = 0
            
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.transformer_epochs} [Val]')
                for market_data, stock_data, market_target, stock_target, target in val_bar:
                    market_data = market_data.to(self.device)
                    stock_data = stock_data.to(self.device)
                    market_target = market_target.to(self.device)
                    stock_target = stock_target.to(self.device)
                    target = target.to(self.device)

                    market_output = self.market_model(market_data)
                    stock_output = self.stock_model(stock_data)
                    
                    market_loss = criterion(market_output, market_target)
                    stock_loss = criterion(stock_output, stock_target)
                    
                    val_market_loss += market_loss.item()
                    val_stock_loss += stock_loss.item()
                    
                    val_bar.set_postfix({
                        'mkt_loss': market_loss.item(),
                        'stk_loss': stock_loss.item()
                    })
            
            avg_val_market = val_market_loss / len(val_loader)
            avg_val_stock = val_stock_loss / len(val_loader)
            val_market_losses.append(avg_val_market)
            val_stock_losses.append(avg_val_stock)
            
            # Update schedulers
            scheduler_market.step(avg_val_market)
            scheduler_stock.step(avg_val_stock)
            
            # Save best models
            if avg_val_market < best_market_loss:
                best_market_loss = avg_val_market
                torch.save(self.market_model.state_dict(), f'./artifacts/{self.name}_market_model.pth')
                print(f"  ✓ Saved best market model (val_loss: {avg_val_market:.6f})")
            
            if avg_val_stock < best_stock_loss:
                best_stock_loss = avg_val_stock
                torch.save(self.stock_model.state_dict(), f'./artifacts/{self.name}_stock_model.pth')
                print(f"  ✓ Saved best stock model (val_loss: {avg_val_stock:.6f})")
            
            # Print epoch summary
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1}/{self.transformer_epochs}")
                print(f"  Train - Market: {avg_train_market:.6f}, Stock: {avg_train_stock:.6f}")
                print(f"  Val   - Market: {avg_val_market:.6f}, Stock: {avg_val_stock:.6f}")
                print(f"  LR    - Market: {optimizer_market.param_groups[0]['lr']:.6f}, Stock: {optimizer_stock.param_groups[0]['lr']:.6f}")
                print("-" * 60)
        
        self.market_model.eval()
        self.stock_model.eval()
        
        test_market_losses=[]
        test_stock_losses=[]
        test_market_loss = 0
        test_stock_loss = 0
        
        with torch.no_grad():
            for market_data, stock_data, market_target, stock_target, target in test_loader:
                market_data = market_data.to(self.device)
                stock_data = stock_data.to(self.device)
                market_target = market_target.to(self.device)
                stock_target = stock_target.to(self.device)
                target = target.to(self.device)

                market_output = self.market_model(market_data)
                stock_output = self.stock_model(stock_data)
                
                market_loss = criterion(market_output, market_target)
                stock_loss = criterion(stock_output, stock_target)
                
                test_market_loss += market_loss.item()
                test_stock_loss += stock_loss.item()
                
        
        avg_test_market = test_market_loss / len(test_loader)
        avg_test_stock = test_stock_loss / len(test_loader)
        test_market_losses.append(avg_test_market)
        test_stock_losses.append(avg_test_stock)
        

        print("\nTraining completed!\n")
        print("=" * 60)
        print(f"\nTest Market Loss: {avg_test_market:.6f}")
        print(f"Test Stock Loss: {avg_test_stock:.6f}")

        self.market_model.load_state_dict(torch.load(f'./artifacts/{self.name}_market_model.pth', map_location=self.device))    
        self.stock_model.load_state_dict(torch.load(f'./artifacts/{self.name}_stock_model.pth', map_location=self.device)) 
     
    # ── Phase 2: NN pre-training ──────────────
    def pretrain_nn(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        print("── Phase 2: NN surrogate pre-training ──")

        optimizer = torch.optim.Adam([
            {'params': self.aggregator.parameters(), 'lr': self.lr},
            {'params': self.fusion.parameters(), 'lr': self.lr},
            {'params': self.head.parameters(), 'lr': self.lr},
            {'params': self.market_model.parameters(), 'lr': self.lr * 0.1},
            {'params': self.stock_model.parameters(), 'lr': self.lr * 0.1},
        ])

        criterion = nn.BCEWithLogitsLoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_loss = float('inf')

        train_losses = []
        val_losses = []

        for epoch in range(self.nn_epochs):
            # ───────── TRAIN ─────────
            self.aggregator.train()
            self.fusion.train()
            self.head.train()
            self.market_model.train()
            self.stock_model.train()

            train_loss = 0

            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.nn_epochs} [Train]')

            for market_data, stock_data, _, _, target in train_bar:
                market_data = market_data.to(self.device)
                stock_data = stock_data.to(self.device)
                target = target.to(self.device)

                B = stock_data.shape[0]

                # Forward
                market_output = self.market_model(market_data)   # (B, 11, 18)
                stock_output = self.stock_model(stock_data)      # (B, 29, 18)

                flatten_market_output = market_output.view(B, -1)
                market_vec = self.aggregator(market_output)      # (B, 18)

                preds_list, targets_list = [], []

                for stock_idx in range(STOCK_INDICES):
                    s_vec = stock_output[:, stock_idx, :]
                    ls_vec = stock_data[:, stock_idx, -1, :]
                    fused = self.fusion(s_vec, market_vec)

                    input_vec = torch.cat(
                        [ls_vec, s_vec, market_vec, fused, flatten_market_output],
                        dim=1
                    )

                    pred = self.head(input_vec)
                    preds_list.append(pred)
                    targets_list.append(target[:, stock_idx, :])

                preds = torch.cat(preds_list, dim=0)
                targets = torch.cat(targets_list, dim=0)

                loss = criterion(preds, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.aggregator.parameters()) +
                    list(self.fusion.parameters()) +
                    list(self.head.parameters()),
                    max_norm=1.0
                )
                optimizer.step()

                train_loss += loss.item()

                train_bar.set_postfix({'loss': loss.item()})

            avg_train = train_loss / len(train_loader)
            train_losses.append(avg_train)

            # ───────── VALIDATION ─────────
            self.aggregator.eval()
            self.fusion.eval()
            self.head.eval()
            self.market_model.eval()
            self.stock_model.eval()

            val_loss = 0

            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.nn_epochs} [Val]')

                for market_data, stock_data, _, _, target in val_bar:
                    market_data = market_data.to(self.device)
                    stock_data = stock_data.to(self.device)
                    target = target.to(self.device)

                    B = stock_data.shape[0]

                    market_output = self.market_model(market_data)
                    stock_output = self.stock_model(stock_data)

                    flatten_market_output = market_output.view(B, -1)
                    market_vec = self.aggregator(market_output)

                    preds_list, targets_list = [], []

                    for stock_idx in range(STOCK_INDICES):
                        s_vec = stock_output[:, stock_idx, :]
                        ls_vec = stock_data[:, stock_idx, -1, :]
                        fused = self.fusion(s_vec, market_vec)

                        input_vec = torch.cat(
                            [ls_vec, s_vec, market_vec, fused, flatten_market_output],
                            dim=1
                        )

                        pred = self.head(input_vec)
                        preds_list.append(pred)
                        targets_list.append(target[:, stock_idx, :])

                    preds = torch.cat(preds_list, dim=0)
                    targets = torch.cat(targets_list, dim=0)

                    loss = criterion(preds, targets)
                    val_loss += loss.item()

                    val_bar.set_postfix({'loss': loss.item()})

            avg_val = val_loss / len(val_loader)
            val_losses.append(avg_val)

            # Scheduler step
            scheduler.step(avg_val)

            # Save best model
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save({
                    'aggregator': self.aggregator.state_dict(),
                    'fusion': self.fusion.state_dict(),
                    'head': self.head.state_dict(),
                    'market_model': self.market_model.state_dict(),
                    'stock_model': self.stock_model.state_dict(),
                }, f'./artifacts/{self.name}_nn_model.pth')

                print(f"  ✓ Saved best NN model (val_loss: {avg_val:.6f})")

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1}/{self.nn_epochs}")
                print(f"  Train Loss: {avg_train:.6f}")
                print(f"  Val   Loss: {avg_val:.6f}")
                print(f"  LR         : {optimizer.param_groups[0]['lr']:.6f}")
                print("-" * 60)

        print("\nNN pretraining completed!\n")

        # Load best model
        checkpoint = torch.load(f'./artifacts/{self.name}_nn_model.pth', map_location=self.device)
        self.aggregator.load_state_dict(checkpoint['aggregator'])
        self.fusion.load_state_dict(checkpoint['fusion'])
        self.head.load_state_dict(checkpoint['head'])
        self.market_model.load_state_dict(checkpoint['market_model'])
        self.stock_model.load_state_dict(checkpoint['stock_model'])

    # ── Phase 3: XGBoost training ─────────────
    def train_xgboost(self, loader: DataLoader) -> None:
        """
        Extract frozen NN features for the full dataset, then fit XGBoost classifier.
        """
        print("── Phase 2: Collecting features for XGBoost ──")
        all_features, all_targets = [], []

        self.market_model.eval()
        self.stock_model.eval()
        self.aggregator.eval()
        self.fusion.eval()

        for market_data, stock_data, market_target, stock_target, target in loader:
            market_data = market_data.to(self.device)
            stock_data = stock_data.to(self.device)
            target = target.to(self.device)


            with torch.no_grad():
                market_output = self.market_model(market_data)
                stock_output = self.stock_model(stock_data)
                feats = self._extract_features(stock_output, market_output,stock_data)  # (B*29, 18)
                tgts = target.cpu().numpy().reshape(-1)  # Flatten to (B*29,)

            all_features.append(feats)
            all_targets.append(tgts)

        X_xgb = np.vstack(all_features)
        y_xgb = np.concatenate(all_targets)

        print(f"  XGBoost input: X={X_xgb.shape}, y={y_xgb.shape}")
        print(f"  Class distribution: 0={np.sum(y_xgb==0)}, 1={np.sum(y_xgb==1)}")
        
        print("── Fitting XGBoost Classifier ──")
        
        # FIX: Use XGBClassifier for binary classification
        self.xgb_params.update({
            "objective": "binary:logistic",  # Binary classification
            "eval_metric": "logloss",        # Appropriate metric
        })
        
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.xgb_model.fit(
            X_xgb, y_xgb,
            verbose=50,
        )
        print("  XGBoost training complete.")

    # ── Full pipeline ─────────────────────────
    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None, 
            use_optuna: bool = False, n_trials: int = 50):
        """
        Full pipeline: pre-train NN → fit XGBoost with Optuna → optional validation.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            use_optuna: Whether to use Optuna for hyperparameter optimization
            n_trials: Number of Optuna trials (if use_optuna=True)
        """
        self.pretrain_transformers(train_loader, val_loader)
        self.pretrain_nn(train_loader,val_loader)
        
        if use_optuna:
            self.train_xgboost_with_optuna(train_loader, n_trials=n_trials)
        else:
            self.train_xgboost(train_loader)  # Your original method

        if val_loader is not None:
            self.evaluate(val_loader, split="val")

    # ── Inference ────────────────────────────
    def predict(self, market_data, stock_data):
        market_output = self.market_model(market_data.to(self.device))
        stock_output  = self.stock_model(stock_data.to(self.device))

        feats = self._extract_features(stock_output, market_output,stock_data)
        flat  = self.xgb_model.predict(feats)

        B = stock_data.shape[0]
        return flat.reshape(B, STOCK_INDICES)

    # ── Evaluation ───────────────────────────
    def evaluate(self, loader: DataLoader, split: str = "test") -> dict[str, float]:
        """
        Compute metrics over a DataLoader split for binary classification.
        """
        assert self.xgb_model is not None, "Call fit() first."
        all_preds, all_targets = [], []

        for market_data, stock_data, market_target, stock_target, target in loader:
            market_data = market_data.to(self.device)
            stock_data = stock_data.to(self.device)
            
            # Get probability predictions
            preds_proba = self.predict(market_data, stock_data)  # Returns probabilities
            targets = target.numpy().reshape(-1, STOCK_INDICES)
            
            all_preds.append(preds_proba)
            all_targets.append(targets)

        y_pred_proba = np.vstack(all_preds).ravel()
        y_pred_class = (y_pred_proba > 0.5).astype(int)
        y_true = np.vstack(all_targets).ravel()
        
        # Calculate classification metrics
        
        accuracy = accuracy_score(y_true, y_pred_class)
        precision = precision_score(y_true, y_pred_class, zero_division=0)
        recall = recall_score(y_true, y_pred_class, zero_division=0)
        f1 = f1_score(y_true, y_pred_class, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        print(f"  [{split}] Accuracy: {accuracy:.4f}")
        print(f"  [{split}] Precision: {precision:.4f}")
        print(f"  [{split}] Recall: {recall:.4f}")
        print(f"  [{split}] F1-Score: {f1:.4f}")
        print(f"  [{split}] AUC-ROC: {auc:.4f}")
        
        report=classification_report(y_true, y_pred_class, zero_division=0)
        print(f"\nClassification Report:\n{report}")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }

    # ── Persistence ──────────────────────────
    def save(self, path_prefix: str = "market_pipeline"):
        """Save NN weights and XGBoost model."""
        torch.save({
            "aggregator": self.aggregator.state_dict(),
            "fusion":     self.fusion.state_dict(),
            "head":       self.head.state_dict(),
            "market_model": self.market_model.state_dict(),
            "stock_model": self.stock_model.state_dict(),
        }, f"./artifacts/{path_prefix}_nn.pt")

        if self.xgb_model is not None:
            self.xgb_model.save_model(f"./artifacts/{path_prefix}_xgb.json")

        print(f"  Saved to ./artifacts/{path_prefix}_nn.pt + ./artifacts/{path_prefix}_xgb.json")

    def load(self, path_prefix: str = "market_pipeline"):
        """Load NN weights and XGBoost model."""
        ckpt = torch.load(f"{path_prefix}_nn.pt", map_location=self.device)

        self.aggregator.load_state_dict(ckpt["aggregator"])
        self.fusion.load_state_dict(ckpt["fusion"])
        self.head.load_state_dict(ckpt["head"])
        self.market_model.load_state_dict(ckpt["market_model"])
        self.stock_model.load_state_dict(ckpt["stock_model"])

        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.xgb_model.load_model(f"{path_prefix}_xgb.json")
        print("  Models loaded.")

    def train_xgboost_with_optuna(self, loader: DataLoader, n_trials: int = 50, cv_folds: int = 5) -> None:
        """
        Extract frozen NN features and optimize XGBoost hyperparameters with Optuna.
        
        Args:
            loader: DataLoader for training data
            n_trials: Number of Optuna trials
            cv_folds: Number of cross-validation folds
        """
        print("── Phase 2: Collecting features for XGBoost ──")
        all_features, all_targets = [], []

        self.market_model.eval()
        self.stock_model.eval()
        self.aggregator.eval()
        self.fusion.eval()

        # Extract features
        for market_data, stock_data, market_target, stock_target, target in loader:
            market_data = market_data.to(self.device)
            stock_data = stock_data.to(self.device)
            target = target.to(self.device)


            with torch.no_grad():
                market_output = self.market_model(market_data)
                stock_output = self.stock_model(stock_data)
                feats = self._extract_features(stock_output, market_output,stock_data)  # (B*29, 18)
                tgts = target.cpu().numpy().reshape(-1)  # Flatten to (B*29,)

            all_features.append(feats)
            all_targets.append(tgts)

        X_xgb = np.vstack(all_features)
        y_xgb = np.concatenate(all_targets)

        print(f"  XGBoost input: X={X_xgb.shape}, y={y_xgb.shape}")
        print(f"  Class distribution: 0={np.sum(y_xgb==0)}, 1={np.sum(y_xgb==1)}")
        
        # Handle class imbalance if needed
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_xgb), y=y_xgb)
        scale_pos_weight = class_weights[1] / class_weights[0] if class_weights[0] > 0 else 1.0
        print(f"  Scale positive weight: {scale_pos_weight:.4f}")
        
        print("── Optuna Hyperparameter Optimization ──")
        
        def objective(trial):
            """Objective function for Optuna optimization."""
            
            # Hyperparameter search space
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'verbosity': 0,
                'use_label_encoder': False,
                'seed': 42,
                
                # Tree-specific parameters
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                
                # Boosting parameters
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                
                # Regularization
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
                
                # Sampling
                'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
            }
            
            # Cross-validation
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in skf.split(X_xgb, y_xgb):
                X_train_fold = X_xgb[train_idx]
                y_train_fold = y_xgb[train_idx]
                X_val_fold = X_xgb[val_idx]
                y_val_fold = y_xgb[val_idx]
                
                # Train XGBoost
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False
                )
                
                # Predict and evaluate
                y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                auc_score = roc_auc_score(y_val_fold, y_pred_proba)
                cv_scores.append(auc_score)
            
            return np.mean(cv_scores)
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',  # Maximize AUC
            sampler=TPESampler(seed=42),
            study_name='xgboost_optimization'
        )
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        best_auc = study.best_value
        
        print(f"\n  Best AUC (CV): {best_auc:.4f}")
        print(f"  Best parameters:")
        for key, value in best_params.items():
            print(f"    {key}: {value}")
        
        # Add fixed parameters
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'use_label_encoder': False,
            'seed': 42,
            'scale_pos_weight': scale_pos_weight,
        })
        
        # Train final model with best parameters on full dataset
        print("\n── Training final XGBoost model with best parameters ──")
        self.xgb_model = xgb.XGBClassifier(**best_params)
        self.xgb_model.fit(
            X_xgb, y_xgb,
            verbose=False,
        )
        
        # Store optimization results
        self.optuna_study = study
        self.best_xgb_params = best_params
        
        print("  XGBoost training complete with optimized hyperparameters!")
        
        # Optional: Plot optimization results
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_optimization_history, plot_param_importances
            
            fig1 = plot_optimization_history(study)
            fig1.show()
            
            fig2 = plot_param_importances(study)
            fig2.show()
        except:
            pass


testing=MarketPipeline(name="Try")
testing.fit(train_loader,val_loader)
testing.evaluate(test_loader,split="test")