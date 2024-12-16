import torch
import torch.nn as nn
import torch.nn.functional as F

#############################
# Losses
#############################

class GPLoss(nn.Module):
    def __init__(self):
        super(GPLoss, self).__init__()
        self.trace = SPLoss()

  
    def get_image_gradients(self,input):        
        f_v_1 = F.pad(input,(0,-1,0,0))
        f_v_2 = F.pad(input,(-1,0,0,0))
        f_v = f_v_1-f_v_2

        f_h_1 = F.pad(input,(0,0,0,-1))
        f_h_2 = F.pad(input,(0,0,-1,0))
        f_h = f_h_1-f_h_2

        return f_v, f_h

    def __call__(self, input, reference):
        ## comment these lines when you inputs and outputs are in [0,1] range already
        input = (input+1)/2
        reference = (reference+1)/2
       
        input_v,input_h = self.get_image_gradients(input)
        ref_v, ref_h = self.get_image_gradients(reference)

        trace_v = self.trace(input_v,ref_v)
        trace_h = self.trace(input_h,ref_h)
        return trace_v + trace_h

class CPLoss(nn.Module):
    def __init__(self,rgb=True,yuv=True,yuvgrad=True):
        super(CPLoss, self).__init__()
        self.rgb = rgb
        self.yuv = yuv
        self.yuvgrad = yuvgrad
        self.trace = SPLoss()
        self.trace_YUV = SPLoss()
    
    def get_image_gradients(self,input):       
        f_v_1 = F.pad(input,(0,-1,0,0))
        f_v_2 = F.pad(input,(-1,0,0,0))
        f_v = f_v_1-f_v_2

        f_h_1 = F.pad(input,(0,0,0,-1))
        f_h_2 = F.pad(input,(0,0,-1,0))
        f_h = f_h_1-f_h_2

        return f_v, f_h

    def to_YUV(self,input):
        return torch.cat((0.299*input[:,0,:,:].unsqueeze(1)+0.587*input[:,1,:,:].unsqueeze(1)+0.114*input[:,2,:,:].unsqueeze(1),\
         0.493*(input[:,2,:,:].unsqueeze(1)-(0.299*input[:,0,:,:].unsqueeze(1)+0.587*input[:,1,:,:].unsqueeze(1)+0.114*input[:,2,:,:].unsqueeze(1))),\
         0.877*(input[:,0,:,:].unsqueeze(1)-(0.299*input[:,0,:,:].unsqueeze(1)+0.587*input[:,1,:,:].unsqueeze(1)+0.114*input[:,2,:,:].unsqueeze(1)))),dim=1)


    def __call__(self, input, reference):
        ## comment these lines when you inputs and outputs are in [0,1] range already
        input = (input+1)/2
        reference = (reference+1)/2
        total_loss= 0
        if self.rgb:
            total_loss += self.trace(input,reference)
        if self.yuv:
            input_yuv = self.to_YUV(input)
            reference_yuv = self.to_YUV(reference)
            total_loss += self.trace(input_yuv,reference_yuv)
        if self.yuvgrad:
            input_v,input_h = self.get_image_gradients(input_yuv)
            ref_v,ref_h = self.get_image_gradients(reference_yuv)

            total_loss +=  self.trace(input_v,ref_v)
            total_loss +=  self.trace(input_h,ref_h)

        return total_loss

class SPL_ComputeWithTrace(nn.Module):
    """
    Slow implementation of the trace loss using the same formula as stated in the paper.
    """
    def __init__(self,weight = [1.,1.,1.]):
        super(SPL_ComputeWithTrace, self).__init__()
        self.weight = weight

    def __call__(self, input, reference):
        a = 0
        b = 0
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                a += torch.trace(torch.matmul(F.normalize(input[i,j,:,:],p=2,dim=1),torch.t(F.normalize(reference[i,j,:,:],p=2,dim=1))))/input.shape[2]*self.weight[j]
                b += torch.trace(torch.matmul(torch.t(F.normalize(input[i,j,:,:],p=2,dim=0)),F.normalize(reference[i,j,:,:],p=2,dim=0)))/input.shape[3]*self.weight[j]
        a = -torch.sum(a)/input.shape[0]
        b = -torch.sum(b)/input.shape[0]
        return a+b

# refer to "Content and colour distillation for learning image translations with the spatial profile loss"

class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss, self).__init__()
        

    def __call__(self, input, reference):
        a = torch.sum(torch.sum(F.normalize(input, p=2, dim=2) * F.normalize(reference, p=2, dim=2),dim=2, keepdim=True))
        b = torch.sum(torch.sum(F.normalize(input, p=2, dim=3) * F.normalize(reference, p=2, dim=3),dim=3, keepdim=True))
        return -(a + b) / input.size(2)
