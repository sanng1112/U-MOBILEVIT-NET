import argparse
import torch
from torch import nn, Tensor
from typing import Optional, Union, Tuple, List, Any
import torch.optim as optim


from cv_nets.utils.config_helper import get_param



from cv_nets.layers.activation import build_activation_layer
from cv_nets.layers.normalization import build_normalization_layer
from cv_nets.layers.pooling import build_pooling_layer
from cv_nets.layers import *




class DoubleConvBlock(nn.Module):
    def __init__(self, opts: Any, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            opts=opts
        )
        self.norm1 = build_normalization_layer(opts, num_features=out_channels)
        self.act1 = build_activation_layer(opts)
        self.conv2 = Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            opts=opts
        )
        self.norm2 = build_normalization_layer(opts, num_features=out_channels)
        self.act2 = build_activation_layer(opts)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class UNetLite(nn.Module):
    def __init__(self, opts: Any, num_classes: int = 1) -> None:
        super().__init__()
        features = [32, 64, 128, 256]
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        in_ch = 3 
        for feat in features:
            self.encoder_blocks.append(DoubleConvBlock(opts, in_ch, feat))
            self.downsample_layers.append(
                Conv2d(in_channels=feat, out_channels=feat, kernel_size=2, stride=2, padding=0, opts=opts)
            )
            in_ch = feat
        self.bottleneck = DoubleConvBlock(opts, features[-1], features[-1] * 2) 

        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        for feat in reversed(features):
            self.upsample_layers.append(
                ConvTranspose2d(in_channels=feat * 2, out_channels=feat, kernel_size=2, stride=2, padding=0, opts=opts)
            )
            self.decoder_blocks.append(DoubleConvBlock(opts, feat * 2, feat))

        self.final_conv = Conv2d(in_channels=features[0], out_channels=num_classes, kernel_size=1, stride=1, padding=0, opts=opts)

    def forward(self, x: Tensor) -> Tensor:
        skip_connections: List[Tensor] = []
        
        for i in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[i](x)
            skip_connections.append(x)  
            x = self.downsample_layers[i](x)
            
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        
        for i in range(len(self.decoder_blocks)):
            x = self.upsample_layers[i](x)     
            skip_connection = skip_connections[i] 
            x_cat = torch.cat((skip_connection, x), dim=1)
            x = self.decoder_blocks[i](x_cat)   
            
        return self.final_conv(x)



def check_model_convergence(model, device='cpu'):
    print(f"Đưa mô hình lên {device.upper()} và bắt đầu kiểm tra...")
    model.to(device)
    model.train() 

    batch_size = 2
    x = torch.randn(batch_size, 3, 320, 320, device=device) 
    
    y_true = torch.zeros((batch_size, 1, 320, 320), dtype=torch.float32, device=device)
    y_true[:, :, 100:220, 100:220] = 1.0  
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Bắt đầu ép mô hình overfit trên 1 batch dữ liệu duy nhất:\n")

    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()     

        y_pred = model(x)         
        
        loss = criterion(y_pred, y_true) 
        
        loss.backward()           
        optimizer.step()           

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1:3d}/{epochs}] - Loss: {loss.item():.6f}")

    print("\n=> Kiểm tra hoàn tất!")
    return loss.item()

# if __name__ == "__main__":
#     class DummyOpts:
#         pass
#     opts = DummyOpts()

#     model = UNetLite(opts=opts, num_classes=1)
#     print(model)
    
#     device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
#     final_loss = check_model_convergence(model, device)



