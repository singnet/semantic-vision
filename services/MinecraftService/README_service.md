# How to use service via UI or via snet

### Via UI

In the UI you need to choose Method Name (currently there is only one - getMinecraftiziedImage), Model Name (currently
it's UGATIT or cycle_gan) and dataset (currently only minecraft_landscapes is availiable). Then, you need to upload 
input image and click "invoke" button. After that, result image and input image will be showed after some calculations.

### Via snet

Here you need to make a snet request via terminal. Here is an example of what sort of request could be sent

    snet client call snet minecraftizing-service default_group getMinecraftiziedImage '{ "file@input_image": "1.jpg", 
    "network_name": "UGATIT", "dataset": "minecraft_landscapes" }'

As a result, you will receive output image "output" and "status" message. Output image is an image, converted to base64.
In client.py is an example of how to convert it back and save as jpg using python.