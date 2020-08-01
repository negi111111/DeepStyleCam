# DeepStyleCam: A Real-Time Style Transfer App on iOS

In this demo, we present a very fast CNN-based style transfer system running on normal iPhones. The proposed app can transfer multiple pre-trained styles to the video stream captured from the builtin camera of an iPhone around 140ms (7fps). We extended the network
proposed as a real-time neural style transfer network by Johnson et al. [1] so that the network can learn multiple styles at the same time. In addition, we modified the CNN network so that the amount of computation is reduced one tenth compared to the original network. The very fast mobile implementation of the app are based on our paper [2] which describes several
new ideas to implement CNN on mobile devices efficiently. Figure 1 shows an example usage of DeepStyleCam which is running on an iPhone SE.

<img src="https://github.com/negi111111/DeepStyleCam/blob/master/data/deepstylecam.gif"/>

## Demo

- Youtube link is [here](https://youtu.be/ZwfBBYy5I10)
  - Multi Style Transfer Mobile Implementation with Chainer running on iPad Pro 12.9
- Youtube link is [here](https://youtu.be/Ut5WYGi5yRU)
  - DeepStyleCam: A Real-time Multi-Style Transfer App on iOS
- Youtube link is [here](https://youtu.be/HMCJXejuJ9Q)
  - The Prototype

## Dependencies

- C for Neural Network Engine
- Objective-C for iOS Programming
- OpenCV
- iOS >= 11.0
- Xcode >= 9.0

## Install

1.  `git clone https://github.com/negi111111/DeepStyleCam.git`
2.  `cd ./DeepStyleCam`
3.  Compile project with Xcode

## Citation

If you use this app in a publication, a link to or citation of this repository would be appreciated.

```
@InProceedings{tann17,
  author="Tanno, R. and Yanai, K.",
  title="DeepStyleCam: A Real-time Style Transfer App on iOS",
  booktitle="Proc. of International MultiMedia Modeing Conference (MMM)",
  year="2017"
}
```

## License

MIT. Copyright (c) 2017 Ryosuke Tanno
