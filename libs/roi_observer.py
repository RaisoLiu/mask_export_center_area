import torch

class DinoV2latentGen:
    def __init__(self, model_cfg):
        self.model = torch.hub.load('dinov2/', model_cfg['name'], source='local', pretrained=False)
        self.model.load_state_dict(torch.load(model_cfg['path']))
        device = model_cfg['device']
        self.n_feature = self.model.embed_dim

        if len(device) == 0:
            self.cal_device = 'cpu'
            if torch.cuda.is_available():
                self.cal_device = 'cuda' 
            if torch.backends.mps.is_available():
                self.cal_device = 'mps'
        else:
            self.cal_device = device
            
        print("Device: ", self.cal_device)
        
    def batch_run(self, X):
        
        if type(X) == list:
            X = torch.stack(X)
        return self.run(X)
    
    def single_run(self, x):
        X = torch.unsqueeze(x, 0)
        return self.run(X)
        
    def run(self, X):
        self.model.eval()
        self.model.to(self.cal_device)
        with torch.no_grad():
            X = X.to(self.cal_device)
            result = self.model.forward_features(X)
        
        return result['x_norm_patchtokens'].detach().cpu().numpy()
    
    def __del__(self):
        del self.model
        torch.cuda.empty_cache()
   