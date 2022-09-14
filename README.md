# NeuralStyleTransfer
Generate an image in the style of another image. Example: The first two images are inputs, while the third is what a pretrained model outputs.

<img src="https://user-images.githubusercontent.com/62298758/190207476-9d0e03b3-80a7-4296-8e4e-782d2a620e3a.jpg" width="400" height="300"/>
<img src="https://user-images.githubusercontent.com/62298758/190207574-cc3514c4-d3c0-455e-a21b-e0f147d4f937.jpg" width="400" height="300"/>
<img src="https://user-images.githubusercontent.com/62298758/190207636-3c532a8f-37e5-4100-a32b-cfad662c345e.jpg" width="400" height="300"/>

The way neural style transfer works is by focusing on different layers of the network, for styling and content, as these are the two things the model needs to get right.
The styling would in this case be the colors, sharp edges etc. while the content would be the dog. The model focuses on the style layers when the style input is passed through it, and the content layer(s) when the content input is passed through it.

It then attempts to generate an image that maximizes the similarities between its own and the respective layers of the input images. There are several fast ways to generate an image. This implementation starts off with the content image and then gradually changes it to maximize its similarities on the styling layers.
