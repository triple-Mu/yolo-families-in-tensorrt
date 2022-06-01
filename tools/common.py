import torch
from torch import nn

class End2EndTRT(nn.Module):
    def __init__(self,model,device=None):
        super().__init__()
        model.eval()
        self.model = model
        self.device = device
        self.convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=torch.float32,device=self.device)

    def forward(self,x):
        x = self.model(x)[0]
        box = x[:, :, :4]
        conf = x[:, :, 4:5]
        score = x[:, :, 5:]
        score *= conf
        box @= self.convert_matrix
        return box, score

class End2EndORT(nn.Module):
    def __init__(self,model,device=None,max_wh=7680):
        super().__init__()
        model.eval()
        self.model = model
        self.device = device
        self.max_wh = max_wh
        self.convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=torch.float32,device=self.device)

    def forward(self,x):
        x = self.model(x)[0] # 1,25200,85
        box = x[:, :, :4]
        conf = x[:, :, 4:5]
        score = x[:, :, 5:]
        score *= conf
        box @= self.convert_matrix
        objScore,objCls = score.max(2,keepdim=True)
        dis = objCls.float()*self.max_wh
        nmsbox = box + dis
        objScore1 = objScore.transpose(1,2).contiguous()
        return nmsbox, box, objScore1, objCls,objScore

class PostORT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, selected_indices, boxes, classes, scores):
        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        resBoxes = boxes[X, Y, :]
        resClasses = classes[X, Y, :]
        resScores = scores[X, Y, :]
        X = X.unsqueeze(1)
        X = X.float()
        resClasses = resClasses.float()
        out = torch.concat([X, resBoxes, resClasses, resScores], 1)
        return out
