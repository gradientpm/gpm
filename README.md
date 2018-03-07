Gradient-Domain Photon Density Estimation
=========================================

This is the code release for the paper "Gradient-Domain Photon Density Estimation" in Eurographics 2017. It extends Mitsuba 0.5.0 to include the following rendering techniques: Gradient-Domain Photon Density Estimation(G-PM), 
Gradient-Domain Path Tracing (G-PT) and Gradient-Domain Bidirectional Path Tracing (G-BDPT). 

The code can be compiled on Windows 10 with Visual Studio 2013, and Arch Linux platform. 

Download
========
- [Paper](http://sonhua.github.io/pdf/hua-gpm-eg17.pdf)

- [Scene data and reference images]
  (http://beltegeuse.s3-website-ap-northeast-1.amazonaws.com/research/2017_GPM/comparison/index.html)
  
- Mitsuba dependencies
  + [Windows](https://www.mitsuba-renderer.org/repos/dependencies_windows) 
  + [Mac](https://www.mitsuba-renderer.org/repos/dependencies_macos)
  
License
=======

If you use this code, please consider citing the following works accordingly: 

- Gradient-Domain Photon Density Estimation
```
@article{hua2017gpm,
  title     = {Gradient-Domain Photon Density Estimation},
  author    = {Hua, Binh-Son and Gruson, Adrien and Nowrouzezahrai, Derek and Hachisuka, Toshiya},  
  journal   = {Eurographics},
  year      = {2017},
  publisher = {The Eurographics Association},
}
```

- Gradient-Domain Bidirectional Path Tracing 
```
@Article{manzi2015gbdpt,
  author    = {Manzi, Marco and Kettunen, Markus and Aittala, Miika and Lehtinen, Jaakko and Durand, Fr{\'e}do and Zwicker, Matthias},
  title     = {Gradient-domain bidirectional path tracing},
  journal   = {Eurographics Symposium on Rendering},
  year      = {2015},
}
```

- Gradient-Domain Path Tracing 
```
@article{kettunen2015gpt,
  author    = {Kettunen, Markus and Manzi, Marco and Aittala, Miika and Lehtinen, Jaakko and Durand, Fr{\'e}do and Zwicker, Matthias},
  title     = {Gradient-domain path tracing},
  journal   = {ACM Transactions on Graphics (TOG)},
  year      = {2015},
  volume    = {34},
  number    = {4},
}
```

- Gradient-Domain Metropolis Light Transport 
```
@article{lehtinen2013gmlt,
  author    = {Lehtinen, Jaakko and Karras, Tero and Laine, Samuli and Aittala, Miika and Durand, Fr{\'e}do and Aila, Timo},
  title     = {Gradient-domain metropolis light transport},
  journal   = {ACM Transactions on Graphics (TOG)},
  year      = {2013},
  volume    = {32},
  number    = {4},
}
```

- Mitsuba Renderer
```
@misc{jakob2010mitsuba,
  author = {Jakob, Wenzel},
  title  = {Mitsuba Renderer},
  year   = {2010},
  note   = {http://www.mitsuba-renderer.org},
}
```

This source code includes the following open source implementations:

- Screened Poisson reconstruction code from NVIDIA, released under the new BSD license.
- Mitsuba 0.5.0 by Wenzel Jakob, released under the GNU General Public License (version 3).

Contact
=======

Please feel free to email `binhson.hua[at]gmail.com` or `adrien.gruson[at]gmail.com` for questions regarding the code. 

