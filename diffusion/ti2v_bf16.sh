python run_wan22_ti2v.py \
    --model-path /home/weights/Wan2.2-TI2V-5B-Diffusers \
    --image-path gyro.jpg \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --height 704 --width 1280 \
    --num-gpus 1 \
    --num-frames 81 \
    --num-inference-steps 40 \
    --output-dir ./outputs_bf16
