# EIP_Session4

Look at [Success02_withgradcam](https://github.com/abhinavdayal/EIP_Session4/blob/master/Succes02_withGradcam.ipynb) for first success and GradCam.

It is using the standard Resnet20 provided in example code with slightly tweaked learning rate, no cutouts etc. Got 89%+ accuracy.

Finally got Resnet18 to have accuracy of 89%+ [Succes03](https://github.com/abhinavdayal/EIP_Session4/blob/master/Success03.ipynb)

I will post more successes here as I get those.

# Experiments
I did a lot of experimentation with the standard Resnet Architecture in original andmodified paper. But did never reach beyond 87% even using more number of parameters. The above Resnet 20 works great with less than 300K parameters.

Following are my observations:

1. Learning rate tuning is important. While exploring that I read several papers. Some talk about constant rate, some a uniform decay, some exponential decay and some a staicase up down with exponental decay and abrupt increase, which they term as super convergance. I have to experiment several strategies.
2. More parameters does not mean more accuracy. Rather with increase in paramaters it is harder ot train also.
3. Choice of architecture is dependent upon complexity of data. Cfar10 is having only 10 classes so we need not have too many channels. with 64 output channels we are able to get close to 90% accuracy. 


I learnt a lot and it is amazing that the things I was thinking of doing are actually experimented by lots of people and discussed by you in Session 5 Video. Thanks for the wornderful experience.
