from models.youtube.face_evoLVe_PyTorch.align.YTPredictor import YTPredictor
from models.metadata.MDPred3 import MDPred3

class MDPred_v:
    def __init__(self):
        self.yt = YTPredictor(absPath_to_youtube='models/youtube')
        self.m3 = MDPred3()
    
    def get_actor_weight(self,url=None,path=None):
        result = self.yt(url,path)
        cast_list = ["nm"+result[i][0] for i in range(len(result))]
        freq_list = [result[i][1][0] for i in range(len(result))]
        weight_list = [float(itm)/sum(freq_list) for itm in freq_list]
        return cast_list,weight_list

    def __call__(self,url=None,path=None):
        cast_list,weight_list=self.get_actor_weight(url=url,path=path)
        return self.m3(cast_list,weight_list)