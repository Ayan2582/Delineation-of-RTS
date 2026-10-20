import torch 
import torch.nn as nn
import torchvision
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels = 512):
        super(RegionProposalNetwork , self ).__init__()
        self.scales = [128,256,512]
        self.aspect_ratios =[0.5,1,2]
        self.anchor_boxes= len(self.scales)*len(self.aspect_ratio)

        #3*3 conv

        self.rpn_conv = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding =1)

        self.cls_layer = nn.Conv2d(in_channels,
                                   self.anchor_boxes*2,
                                   kernel_size=1,
                                   stride = 1)

        self.bbox_reg_layer = nn.Conv2d(in_channels , 
                                        self.num_anchors*4,
                                        kernel_size=1,
                                        stride = 1)

    def generate_anchors(self ,image , feat):
        grid_h , grid_w = feat.shape[-2:] # h , w of feature map
        image_h , image_w = image.shape[-2:] # h , w of original image

       # Determine how much one feature map pixel corresponds to in the image
        stride_h = torch.tensor(image_h //grid_h,
                                dtype = torch.int64,
                                device = feat.device)
        
        stride_w = torch.tensor(image_w //grid_w,
                                dtype = torch.int64,
                                device = feat.device)

        scales = torch.as_tensor(self.scales,
                                 dtype = feat.dtype,
                                 device = feat.device)

        aspect_ratio = torch.as_tensor(self.aspect_ratios , 
                                       dtype = feat.dtype,
                                       device = feat.device)

        # The below code ensure h/w = aspect_ratios and h*w = 1
        h_ratios = torch.sqrt(self.aspect_ratios)
        w_ratios = 1/h_ratios

        #.view flattens the matrix , 
        ws = (w_ratios[:,None]*scales[None,:]).view(-1) # vector of length 9 , where the first position corresponds to ws required  for one scale and one aspect ratio bbox
        hs = (h_ratios[:, None]*scales[None,:]).view(-1)

        base_anchors = torch.stack([-ws,-hs,ws,hs] , dim = 1)/2

        """[
 [-90.51, -45.25,  90.51,  45.25],   # scale 128, ratio 0.5
 [-181.02,-90.51, 181.02,  90.51],   # scale 256, ratio 0.5
 [-362.04,-181.02,362.04,181.02],    # scale 512, ratio 0.5

 [-64,    -64,     64,     64],      # scale 128, ratio 1
 [-128,   -128,    128,    128],     # scale 256, ratio 1
 [-256,   -256,    256,    256],      # scale 512, ratio 1

 [-45.25, -90.51,  45.25,  90.51],   # scale 128, ratio 2
 [-90.51, -181.02, 90.51, 181.02],   # scale 256, ratio 2
 [-181.02,-362.04,181.02,362.04]     # scale 512, ratio 2
]"""
        base_anchors = base_anchors.round()

        #Get the shifts in x axis(0,1,... W_feat-1)*stride_h

        shifts_x =torch.arange(0 ,grid_w ,
                                device =torch.int32 , 
                                device = feat.device )*stride_w

        shifts_y = torch.arange(0 , grid_h ,
                                dtype = torch.int32,
                                device = feat.device)*stride_h
        
        shifts_y , shifts_x = torch.meshgrid(shifts_y , shifts_x, indexing='ij')

        #(H_feat , W_feat)

        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)
        shifts = torch.stack((shifts_x,shifts_y, shifts_x,shifts_y),dim=1)

        #shifts->(H_feat , W_feat , 4)

        #base_anchors -> (num_foanchors_per_location,4)
        #shifts-> (H_feat *W_feat,4)

        anchors = (shifts.view(-1,1,4) + base_anchors.view(1,-1,4))

        anchors = anchors.reshape(-1,4)

        return anchors

    def forward(self, image, feat, target):
        # Call RPN Layers
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))
        cls_scores = self.cls_layer(rpn_feat)
        box_transform_pred = self.bbox_reg_layer(rpn_feat)

        anchors = self.generate_anchors(image,feat)