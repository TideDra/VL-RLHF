export PYTHONPATH=$PYTHONPATH:/mnt/gozhang/code/VL-RLHF
python inference.py \
        --image-path ../../../data_dir/coco2017/test2017/000000497687.jpg \
        --query "Describe this picture in detail." \
        --text "The image features a group of three friends standing close together, possibly at a bar or a party. Two of the friends are women, and they are both holding glasses with drinks in their hands. One of the women is holding a drink in her hand up close to the camera, while the other has her drink slightly farther away. The third friend is a young man wearing a tie, standing between the two women.\n\nThe group appears to be enjoying themselves, with smiles on their faces, and they are facing the camera. In the background, there is another person partially visible. The scene conveys a sense of camaraderie and fun among the friends."\
        --detector-config "../../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
        --detector-model "../../GroundingDINO/weights/groundingdino_swint_ogc.pth" \
        --api-key "05d739b8fe5141699aa0ab8b8cdacfa2" \
        --end-point "https://test-gpt-api-switzerlan-north.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2023-07-01-preview" \