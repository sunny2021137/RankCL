import numpy as np
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import torch
import torch.optim as optim
from src.loss import RankingAwareContrastiveLoss
# from src.loss import ranking_aware_contrastive_loss
from src.eval import evaluate_metrics
from data.data_loader import data_loader
from src.factory import get_model_loss
from .early_stop import EarlyStopping
from CLOC.loss import OrdinalContrastiveLoss_mm
import torch.nn.functional as F

class RankCLTrainer():

    def __init__(self, config, train_sampler, train_loader, val_loader, test_loader, num_classes, input_dim, logger=None):
        
        self.config = config
        self.logger = logger
        self.num_classes = num_classes
        
        self.model, self.clf_loss_fn = get_model_loss(num_classes, input_dim, config)
        
        
        self.rankcl_loss_fn = RankingAwareContrastiveLoss(num_classes, with_correct_penalty=config['rankcl'].get("with_correct_penalty", True), similarity_metric=config['rankcl'].get("similarity_metric", "cosine"))
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["search_params"]["lr"], weight_decay=self.config["train"]["weight_decay"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_sampler = train_sampler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
                
        self.model.to(self.device)
        self.clf_loss_fn.to(self.device)
        self.rankcl_loss_fn.to(self.device)
        
    
    def train(self):
        if self.config["deep_ordinal_method"] == "CLOC":
            self._train_cloc()
        else:
            self._train_common()
            
    def _train_common(self):
        # early stopping
        early_stopping = EarlyStopping(patience=self.config["train"]["patience"], check_freq=self.config["train"]["val_epoch"])
        best_model_weights = deepcopy(self.model.state_dict())
        best_metric = None
        
        print('Starting training...')
               
        for epoch in range(self.config["train"]["epochs"]): 
            self.model.train()
            
            # create balanced sampler data for this epoch
            X_train, y_train = self.train_sampler.sample_epoch(batch_shuffle=False)            
            train_loader_sample = data_loader(X_train, y_train, batch_size=self.train_sampler.batch_size, num_workers=self.config["num_workers"], shuffle=False)

            loss_epoch, loss_epoch_link, loss_epoch_clf = 0.0, 0.0, 0.0
            n_batches = 0
            
            for data in train_loader_sample:
                self.model.train()
                
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()

                    
                feat_outs, clf_outs = self.model(inputs)

                loss_link = self.rankcl_loss_fn(feat_outs, labels)

                if self.config["deep_ordinal_method"] == "DeepCLMWK":
                    clf_outs = clf_outs.cpu()
                    labels = labels.cpu()
                    
                loss_clf = self.clf_loss_fn(clf_outs, labels)

                
                loss = loss_link * self.config["search_params"]["lambda"] + loss_clf
                

                
                loss.backward()
                self.optimizer.step()
                
                loss_epoch += loss.item()
                loss_epoch_link += loss_link.item()
                loss_epoch_clf += loss_clf.item()
                
                n_batches += 1

            print('Epoch {}/{}\t Loss: {:.8f}\t loss_link: {:.8f}\t loss_clf: {:.8f}'.format(epoch + 1, self.config["train"]["epochs"], loss_epoch / n_batches, loss_epoch_link / n_batches, loss_epoch_clf / n_batches))

            if epoch % self.config["train"]["val_epoch"] == 0 or epoch == self.config["train"]["epochs"] - 1:
                validation_metrics = self.validate_val()

                if self.config["train"]["best_metric_name"] == "QWK":
                    current_metric  = -1*validation_metrics["QWK"]
                elif self.config["train"]["best_metric_name"] == "f1-score(macro avg)":
                    current_metric = -1*validation_metrics["f1-score(macro avg)"]
                elif self.config["train"]["best_metric_name"] == "AMAE":
                    current_metric = validation_metrics["AMAE"]
                else:
                    print("error")
                
                if (best_metric is None) or (current_metric < best_metric):
                    best_metric = current_metric
                    best_model_weights = deepcopy(self.model.state_dict())

                
                early_stopping(validation_metrics["loss_clf"])
                if early_stopping.early_stop:
                    print(f"早停於 epoch {epoch}")    
                    break
                             
        self.model.load_state_dict(best_model_weights)
        print('Finished training.')


    # training 
    def _train_cloc(self):
        
        # INFO: phase 1
        margin_criterion_1 = OrdinalContrastiveLoss_mm(
            n_classes=self.num_classes,
            device=self.device,
            learnable_map= [
                ['learnable', None] for _ in range(self.num_classes - 1)
            ]
        )
        self._train_cloc_phase(1, margin_criterion_1)
        
        # INFO: phase 2
        margins = F.softplus(margin_criterion_1.learnables)
        margin_criterion_2 = OrdinalContrastiveLoss_mm(
            n_classes=self.num_classes,
            device=self.device,
            learnable_map= [
                ['fixed', margins[i]] for i in range(self.num_classes - 1)
            ]
        )
        self._train_cloc_phase(2, margin_criterion_2)            
        

    def _train_cloc_phase(self, phase, margin_criterion):
        print(f"Starting training phase {phase}...")
        # NOTE:
        margin_criterion.to(self.device)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(margin_criterion.parameters()), lr=self.config["search_params"]["lr"], weight_decay=self.config["train"]["weight_decay"])
        
        # early stopping
        early_stopping = EarlyStopping(patience=self.config["train"]["patience"], check_freq=self.config["train"]["val_epoch"])
        best_model_weights = deepcopy(self.model.state_dict())
        best_metric = None
        
        print('Starting training...')
        for epoch in range(self.config["train"]["epochs"]): 
            self.model.train()
            X_train, y_train = self.train_sampler.sample_epoch(batch_shuffle=False)            
            train_loader_sample = data_loader(X_train, y_train, batch_size=self.train_sampler.batch_size, num_workers=self.config["num_workers"], shuffle=False)
            
            loss_epoch, loss_epoch_link, loss_epoch_clf = 0.0, 0.0, 0.0
            n_batches = 0
            
            for data in train_loader_sample:
                self.model.train()
                
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                feat_outs, clf_outs = self.model(inputs)
                
                loss_link = self.rankcl_loss_fn(feat_outs, labels)
                
                if self.config["deep_ordinal_method"] == "DeepCLMWK":
                    clf_outs = clf_outs.cpu()
                    labels = labels.cpu()
                    
                loss_clf = self.clf_loss_fn(clf_outs, labels)
                # NOTE:
                loss = loss_link * self.config["search_params"]["lambda"] + loss_clf + margin_criterion(clf_outs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                loss_epoch += loss.item()
                loss_epoch_link += loss_link.item()
                loss_epoch_clf += loss_clf.item()
                
                n_batches += 1
            
            if epoch % self.config["train"]["val_epoch"] == 0 or epoch == self.config["train"]["epochs"] - 1:
                # NOTE: val loader
                validation_metrics = self.validate_val()
                
                if self.config["train"]["best_metric_name"] == "QWK":
                    current_metric = -1*validation_metrics["QWK"]
                elif self.config["train"]["best_metric_name"] == "f1-score(macro avg)":
                    current_metric = -1*validation_metrics["f1-score(macro avg)"]
                elif self.config["train"]["best_metric_name"] == "AMAE":
                    current_metric = validation_metrics["AMAE"]
                else:
                    print("error")
                
                if (best_metric is None) or (current_metric < best_metric):
                    best_metric = current_metric
                    best_model_weights = deepcopy(self.model.state_dict())
                
                # NOTE:
                if phase == 1:
                    # train loader!!!!!!!!
                    train_metrics = self.validate_train()
                    early_stopping(-1*train_metrics["QWK"])
                else: 
                    early_stopping(validation_metrics["loss_clf"])
                    
                if early_stopping.early_stop:
                    print(f"早停於 epoch {epoch}")    
                    break
                
                
            print('Epoch {}/{}\t Loss: {:.8f}\t loss_link: {:.8f}\t loss_clf: {:.8f}'.format(epoch + 1, self.config["train"]["epochs"], loss_epoch / n_batches, loss_epoch_link / n_batches, loss_epoch_clf / n_batches))
    
        self.model.load_state_dict(best_model_weights)
        print('Finished training.')
        
    def validate_val(self):
        
        idx_label_score = []
        
        loss_clf_epoch = 0.0
        n_batches = 0
        
        self.model.eval()
        with torch.no_grad():
            for data in self.val_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                _, clf_outs = self.model(inputs)
                
                if self.config["deep_ordinal_method"] == "OBDECOC":
                    predictions = self.model.transformer.labels(clf_outs)
                else:
                    predictions = torch.argmax(clf_outs, dim=1)
                
                if self.config["deep_ordinal_method"] == "DeepCLMWK":
                    clf_outs = clf_outs.cpu()
                    labels = labels.cpu()
                loss_clf = self.clf_loss_fn(clf_outs, labels)
                idx_label_score += list(zip(
                                            labels.cpu().data.numpy().tolist(),
                                            predictions.cpu().data.numpy().tolist()
                                            ))
                loss_clf_epoch += loss_clf.item()
                n_batches += 1
        
        labels, preds = zip(*idx_label_score)
        labels = np.array(labels)
        preds = np.array(preds, dtype=int)
        
        unique_labels = range(0, self.num_classes)
        all_dict = evaluate_metrics(labels, preds, self.num_classes)
        all_dict["QWK"] = cohen_kappa_score(labels, preds, weights="quadratic", labels=unique_labels)
        all_dict["loss_clf"]= loss_clf_epoch / n_batches
        print(f'validate f1-score(macro):{all_dict["f1-score(macro avg)"]}, \t QWK:{all_dict["QWK"]}' )
        return all_dict


    def validate_train(self):
        
        idx_label_score = []
        
        loss_clf_epoch = 0.0
        n_batches = 0
        
        self.model.eval()
        with torch.no_grad():
            for data in self.train_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                _, clf_outs = self.model(inputs)
                
                if self.config["deep_ordinal_method"] == "OBDECOC":
                    predictions = self.model.transformer.labels(clf_outs)
                else:
                    predictions = torch.argmax(clf_outs, dim=1)
                
                if self.config["deep_ordinal_method"] == "DeepCLMWK":
                    clf_outs = clf_outs.cpu()
                    labels = labels.cpu()
                loss_clf = self.clf_loss_fn(clf_outs, labels)
                idx_label_score += list(zip(
                                            labels.cpu().data.numpy().tolist(),
                                            predictions.cpu().data.numpy().tolist()
                                            ))
                loss_clf_epoch += loss_clf.item()
                n_batches += 1
        
        labels, preds = zip(*idx_label_score)
        labels = np.array(labels)
        preds = np.array(preds, dtype=int)
        
        unique_labels = range(0, self.num_classes)
        all_dict = evaluate_metrics(labels, preds, self.num_classes)
        all_dict["QWK"] = cohen_kappa_score(labels, preds, weights="quadratic", labels=unique_labels)
        all_dict["loss_clf"]= loss_clf_epoch / n_batches
        print(f'validate f1-score(macro):{all_dict["f1-score(macro avg)"]}, \t QWK:{all_dict["QWK"]}' )
        return all_dict

    def test(self):
        # Testing
        print('Starting testing...')
        idx_label_score = []
        self.model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                _, clf_outs = self.model(inputs)

                if self.config["deep_ordinal_method"] == "OBDECOC":
                    predictions = self.model.transformer.labels(clf_outs)
                else:
                    predictions = torch.argmax(clf_outs, dim=1)
                    
                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(
                                            labels.cpu().detach().numpy().tolist(),
                                            predictions.cpu().detach().numpy().tolist()
                                            ))
        # Compute AUC
        labels, preds = zip(*idx_label_score)
        labels = np.array(labels)
        preds = np.array(preds, dtype=int)
        unique_labels = range(0, self.num_classes)
        all_dict = evaluate_metrics(labels, preds, self.num_classes)
        cm = confusion_matrix(labels, preds, labels=unique_labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        all_dict["confusion_matrix"] = cm_norm
            
        print(f'test f1-score(macro):{all_dict["f1-score(macro avg)"]}, \t QWK:{all_dict["QWK"]}' )
        return all_dict
    
    
class BaselineTrainer():

    def __init__(self, config, train_loader, val_loader, test_loader, num_classes, input_dim, logger=None):
        
        self.config = config
        self.logger = logger
        self.num_classes = num_classes
        
        self.model, self.clf_loss_fn = get_model_loss(num_classes, input_dim, config)
       
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["search_params"]["lr"], weight_decay=self.config["train"]["weight_decay"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
             
        self.model.to(self.device)
        self.clf_loss_fn.to(self.device)
        
    
    def train(self):
        # early stopping
        early_stopping = EarlyStopping(patience=self.config["train"]["patience"], check_freq=self.config["train"]["val_epoch"])
        best_model_weights = deepcopy(self.model.state_dict())
        best_metric = None
        
        print('Starting training...')
               
        for epoch in range(self.config["train"]["epochs"]): 
            self.model.train()
            
            loss_epoch = 0.0
            n_batches = 0
            
            for data in self.train_loader:
                self.model.train()
                
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()

                    
                clf_outs = self.model(inputs)

                if self.config["deep_ordinal_method"] == "DeepCLMWK":
                    clf_outs = clf_outs.cpu()
                    labels = labels.cpu()
                    
                loss = self.clf_loss_fn(clf_outs, labels)

                
                loss.backward()
                self.optimizer.step()
                
                loss_epoch += loss.item()
                n_batches += 1

            print('Epoch {}/{}\t Loss: {:.8f}\t'.format(epoch + 1, self.config["train"]["epochs"], loss_epoch / n_batches))

            if epoch % self.config["train"]["val_epoch"] == 0 or epoch == self.config["train"]["epochs"] - 1:
                validation_metrics = self.validate()

                if self.config["train"]["best_metric_name"] == "QWK":
                    current_metric  = -1*validation_metrics["QWK"]
                elif self.config["train"]["best_metric_name"] == "f1-score(macro avg)":
                    current_metric = -1*validation_metrics["f1-score(macro avg)"]
                elif self.config["train"]["best_metric_name"] == "AMAE":
                    current_metric = validation_metrics["AMAE"]
                else:
                    print("error")
                
                if (best_metric is None) or (current_metric < best_metric):
                    best_metric = current_metric
                    best_model_weights = deepcopy(self.model.state_dict())

                
                early_stopping(validation_metrics["loss_clf"])
                if early_stopping.early_stop:
                    print(f"早停於 epoch {epoch}")    
                    break
                             
        self.model.load_state_dict(best_model_weights)
        print('Finished training.')

    def validate(self):
        
        idx_label_score = []
        
        loss_clf_epoch = 0.0
        n_batches = 0
        
        self.model.eval()
        with torch.no_grad():
            for data in self.val_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                clf_outs = self.model(inputs)
                
                if self.config["deep_ordinal_method"] == "OBDECOC":
                    predictions = self.model.transformer.labels(clf_outs)
                else:
                    predictions = torch.argmax(clf_outs, dim=1)
                
                if self.config["deep_ordinal_method"] == "DeepCLMWK":
                    clf_outs = clf_outs.cpu()
                    labels = labels.cpu()
                loss_clf = self.clf_loss_fn(clf_outs, labels)
                idx_label_score += list(zip(
                                            labels.cpu().data.numpy().tolist(),
                                            predictions.cpu().data.numpy().tolist()
                                            ))
                loss_clf_epoch += loss_clf.item()
                n_batches += 1
        
        labels, preds = zip(*idx_label_score)
        labels = np.array(labels)
        preds = np.array(preds, dtype=int)
        
        unique_labels = range(0, self.num_classes)
        all_dict = evaluate_metrics(labels, preds, self.num_classes)
        all_dict["QWK"] = cohen_kappa_score(labels, preds, weights="quadratic", labels=unique_labels)
        all_dict["loss_clf"]= loss_clf_epoch / n_batches
        print(f'validate f1-score(macro):{all_dict["f1-score(macro avg)"]}, \t QWK:{all_dict["QWK"]}' )
        return all_dict

    def test(self):
        # Testing
        print('Starting testing...')
        idx_label_score = []
        self.model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                clf_outs = self.model(inputs)

                if self.config["deep_ordinal_method"] == "OBDECOC":
                    predictions = self.model.transformer.labels(clf_outs)
                else:
                    predictions = torch.argmax(clf_outs, dim=1)
                    
                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(
                                            labels.cpu().detach().numpy().tolist(),
                                            predictions.cpu().detach().numpy().tolist()
                                            ))
        # Compute AUC
        labels, preds = zip(*idx_label_score)
        labels = np.array(labels)
        preds = np.array(preds, dtype=int)
        unique_labels = range(0, self.num_classes)
        all_dict = evaluate_metrics(labels, preds, self.num_classes)
        cm = confusion_matrix(labels, preds, labels=unique_labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        all_dict["confusion_matrix"] = cm_norm
            
        print(f'test f1-score(macro):{all_dict["f1-score(macro avg)"]}, \t QWK:{all_dict["QWK"]}' )
        return all_dict

class CLOCTrainer():

    def __init__(self, config, train_sampler, train_loader, val_loader, test_loader, num_classes, input_dim, logger=None):
        
        self.config = config
        self.logger = logger
        self.num_classes = num_classes
        
        self.model, self.clf_loss_fn = get_model_loss(num_classes, input_dim, config)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["search_params"]["lr"], weight_decay=self.config["train"]["weight_decay"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_sampler = train_sampler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
                
        self.model.to(self.device)
        self.clf_loss_fn.to(self.device)
        

    def train(self):
        
        # INFO: phase 1
        margin_criterion_1 = OrdinalContrastiveLoss_mm(
            n_classes=self.num_classes,
            device=self.device,
            learnable_map= [
                ['learnable', None] for _ in range(self.num_classes - 1)
            ]
        )
        self._train_cloc_phase(1, margin_criterion_1)
        
        # INFO: phase 2
        margins = F.softplus(margin_criterion_1.learnables)
        margin_criterion_2 = OrdinalContrastiveLoss_mm(
            n_classes=self.num_classes,
            device=self.device,
            learnable_map= [
                ['fixed', margins[i]] for i in range(self.num_classes - 1)
            ]
        )
        self._train_cloc_phase(2, margin_criterion_2)            
        

    def _train_cloc_phase(self, phase, margin_criterion):
        print(f"Starting training phase {phase}...")
        
        margin_criterion.to(self.device)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(margin_criterion.parameters()), lr=self.config["search_params"]["lr"], weight_decay=self.config["train"]["weight_decay"])
        
        # early stopping
        early_stopping = EarlyStopping(patience=self.config["train"]["patience"], check_freq=self.config["train"]["val_epoch"])
        best_model_weights = deepcopy(self.model.state_dict())
        best_metric = None
        
        print('Starting training...')
        for epoch in range(self.config["train"]["epochs"]): 
            self.model.train()
            X_train, y_train = self.train_sampler.sample_epoch(batch_shuffle=False)            
            train_loader_sample = data_loader(X_train, y_train, batch_size=self.train_sampler.batch_size, num_workers=self.config["num_workers"], shuffle=False)
            
            loss_epoch, loss_epoch_margin, loss_epoch_clf = 0.0, 0.0, 0.0
            n_batches = 0
            
            for data in train_loader_sample:
                self.model.train()
                
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                clf_outs = self.model(inputs)
                
                    
                loss_clf = self.clf_loss_fn(clf_outs, labels)
                loss_margin = margin_criterion(clf_outs, labels)
                loss = loss_clf + loss_margin
                
                loss.backward()
                self.optimizer.step()
                
                loss_epoch += loss.item()
                loss_epoch_margin += loss_margin.item()
                loss_epoch_clf += loss_clf.item()
                
                n_batches += 1
            
            if epoch % self.config["train"]["val_epoch"] == 0 or epoch == self.config["train"]["epochs"] - 1:
                validation_metrics = self.validate_val()
                
                if self.config["train"]["best_metric_name"] == "QWK":
                    current_metric = -1*validation_metrics["QWK"]
                elif self.config["train"]["best_metric_name"] == "f1-score(macro avg)":
                    current_metric = -1*validation_metrics["f1-score(macro avg)"]
                elif self.config["train"]["best_metric_name"] == "AMAE":
                    current_metric = validation_metrics["AMAE"]
                else:
                    print("error")
                
                if (best_metric is None) or (current_metric < best_metric):
                    best_metric = current_metric
                    best_model_weights = deepcopy(self.model.state_dict())
            
                if phase == 1:
                    train_metrics = self.validate_train()
                    early_stopping(-1*train_metrics["QWK"])
                else: 
                    early_stopping(validation_metrics["loss_clf"])
                    
                if early_stopping.early_stop:
                    print(f"早停於 epoch {epoch}")    
                    break
                
                
            print('Epoch {}/{}\t Loss: {:.8f}\t loss_margin: {:.8f}\t loss_clf: {:.8f}'.format(epoch + 1, self.config["train"]["epochs"], loss_epoch / n_batches, loss_epoch_margin / n_batches, loss_epoch_clf / n_batches))
    
        self.model.load_state_dict(best_model_weights)
        print('Finished training.')
        
    def validate_val(self):
        
        idx_label_score = []
        
        loss_clf_epoch = 0.0
        n_batches = 0
        
        self.model.eval()
        with torch.no_grad():
            for data in self.val_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                clf_outs = self.model(inputs)
                
                if self.config["deep_ordinal_method"] == "OBDECOC":
                    predictions = self.model.transformer.labels(clf_outs)
                else:
                    predictions = torch.argmax(clf_outs, dim=1)
                
                if self.config["deep_ordinal_method"] == "DeepCLMWK":
                    clf_outs = clf_outs.cpu()
                    labels = labels.cpu()
                loss_clf = self.clf_loss_fn(clf_outs, labels)
                idx_label_score += list(zip(
                                            labels.cpu().data.numpy().tolist(),
                                            predictions.cpu().data.numpy().tolist()
                                            ))
                loss_clf_epoch += loss_clf.item()
                n_batches += 1
        
        labels, preds = zip(*idx_label_score)
        labels = np.array(labels)
        preds = np.array(preds, dtype=int)
        
        unique_labels = range(0, self.num_classes)
        all_dict = evaluate_metrics(labels, preds, self.num_classes)
        all_dict["QWK"] = cohen_kappa_score(labels, preds, weights="quadratic", labels=unique_labels)
        all_dict["loss_clf"]= loss_clf_epoch / n_batches
        print(f'validate f1-score(macro):{all_dict["f1-score(macro avg)"]}, \t QWK:{all_dict["QWK"]}' )
        return all_dict


    def validate_train(self):
        
        idx_label_score = []
        
        loss_clf_epoch = 0.0
        n_batches = 0
        
        self.model.eval()
        with torch.no_grad():
            for data in self.train_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                clf_outs = self.model(inputs)
                
                if self.config["deep_ordinal_method"] == "OBDECOC":
                    predictions = self.model.transformer.labels(clf_outs)
                else:
                    predictions = torch.argmax(clf_outs, dim=1)
                
                if self.config["deep_ordinal_method"] == "DeepCLMWK":
                    clf_outs = clf_outs.cpu()
                    labels = labels.cpu()
                loss_clf = self.clf_loss_fn(clf_outs, labels)
                idx_label_score += list(zip(
                                            labels.cpu().data.numpy().tolist(),
                                            predictions.cpu().data.numpy().tolist()
                                            ))
                loss_clf_epoch += loss_clf.item()
                n_batches += 1
        
        labels, preds = zip(*idx_label_score)
        labels = np.array(labels)
        preds = np.array(preds, dtype=int)
        
        unique_labels = range(0, self.num_classes)
        all_dict = evaluate_metrics(labels, preds, self.num_classes)
        all_dict["QWK"] = cohen_kappa_score(labels, preds, weights="quadratic", labels=unique_labels)
        all_dict["loss_clf"]= loss_clf_epoch / n_batches
        print(f'validate f1-score(macro):{all_dict["f1-score(macro avg)"]}, \t QWK:{all_dict["QWK"]}' )
        return all_dict

    def test(self):
        # Testing
        print('Starting testing...')
        idx_label_score = []
        self.model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                clf_outs = self.model(inputs)

                if self.config["deep_ordinal_method"] == "OBDECOC":
                    predictions = self.model.transformer.labels(clf_outs)
                else:
                    predictions = torch.argmax(clf_outs, dim=1)
                    
                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(
                                            labels.cpu().detach().numpy().tolist(),
                                            predictions.cpu().detach().numpy().tolist()
                                            ))
        # Compute AUC
        labels, preds = zip(*idx_label_score)
        labels = np.array(labels)
        preds = np.array(preds, dtype=int)
        unique_labels = range(0, self.num_classes)
        all_dict = evaluate_metrics(labels, preds, self.num_classes)
        cm = confusion_matrix(labels, preds, labels=unique_labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        all_dict["confusion_matrix"] = cm_norm
            
        print(f'test f1-score(macro):{all_dict["f1-score(macro avg)"]}, \t QWK:{all_dict["QWK"]}' )
        return all_dict
    
    