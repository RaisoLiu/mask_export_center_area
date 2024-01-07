# Check and download aot-ckpt 
if [ ! -f ./ckpt/dinov2_vitb14_reg4_pretrain.pth ]; then
    wget -P ./ckpt https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
else
    echo "dinov2_vitb14_reg4_pretrain.pth already downloaded."
fi


