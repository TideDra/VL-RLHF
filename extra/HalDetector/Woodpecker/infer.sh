export PYTHONPATH=$PYTHONPATH:/mnt/gozhang/code/VL-RLHF
python inference.py \
        --image-path ../../../data_dir/coco2017/test2017/000000497687.jpg \
        --query "Describe this picture in detail." \
        --text "The image features a group of three friends standing close together, possibly at a bar or a party. Two of the friends are women, and they are both holding glasses with drinks in their hands. One of the women is holding a drink in her hand up close to the camera, while the other has her drink slightly farther away. The third friend is a young man wearing a tie, standing between the two women.\n\nThe group appears to be enjoying themselves, with smiles on their faces, and they are facing the camera. In the background, there is another person partially visible. The scene conveys a sense of camaraderie and fun among the friends."\
        --detector-config "../../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
        --detector-model "../../GroundingDINO/weights/groundingdino_swint_ogc.pth" \
        --api-key $api_key \
        --end-point $end_point \
        --api-service "azure" \
        --val-model-path /mnt/gozhang/code/VL-RLHF/ckpts/Qwen-VL-Chat \
        --qa2c-model-path /mnt/gozhang/code/VL-RLHF/ckpts/zerofec-qa2claim-t5-base