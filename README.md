# Tree species recognition from point clouds using deep neural networks

<!-- V okviru magistrska dela se navezujemo na raziskovalno delo Seidel s sodelavci in razširjamo primerjavo na modernejše arhitektur nevronskih mrež, ki lahko posredno ali neposredno obdelajo 3D podatke ali oblakov točk. Za nevronsko mrežo, ki posredno obdeluje 3D podatke smo se odločili za točkovno nevronsko mrežo oz. PointNeXt. Za nevronsko mrežo, ki neposredno obdeluje 3D podatke smo se odločili za navadno konvolucijsko nevronsko mrežo EfficientNetV2. -->

As part of the master’s thesis, we relate to the research work of Seidel et al. [[1](https://doi.org/10.3389/fpls.2021.635440)] and extend the comparison to more modern architectures of neural networks that can directly or indirectly process 3D data or point clouds. For a neural network that indirectly processes 3D data, we opted for a point neural network or PointNeXt. For a neural network that directly processes 3D data, we opted for the conventional convolutional neural network or EfficientNetV2.

<!-- Zaradi zapletov pri pripravi izbranih dveh nevronskih mrež in novosti področja, smo se odločili, da objavimo poenostavljeni implementaciji skupaj z rezultati analize in predobdelave podatkov. Poleg implementacije bo na voljo tudi prosto dostopno magistrsko delo v slovenske jeziku, kjer smo zapisali malo več o nevronskih mrežah, podatkih o drevesnih vrstah, učenju samih nevronskih mrež in zaključnih ugotovitvah. V primeru, da vam je naše zaključno delo prišlo prav, bomo veseli reference. -->

Due to the complications in the preparation of the selected two neural networks and the novelties of the field, we decided to publish the simplified implementation together with the results of the analysis and data preprocessing. In addition to the implementation, there will also be an open-access master’s thesis in Slovenian, where we have written a little more about neural networks, data on tree species, training neural networks and final findings. If our final work was of help, we will be happy of a reference. 😊

## Installation

In our implementation we use pytorch3d version 0.7.5, which requires Python version from 3.8 up to 3.10. You could use a newer version of Python, but it will require some additional libraries to compile, like cuda-toolkit.

<!-- Pri naši implementaciji uporabljamo različico pytorch3d 0.7.5, ki zahteva različico Python od 3.8 do 3.10. Lahko uporabite novejšo različico Pythona, vendar bo za prevajanje potrebnih nekaj dodatnih knjižnic, npr. cuda-toolkit. -->

First, make sure you have installed one of the latest drivers; otherwise you can have problems with GPU availability with the PyTorch library. You should pay attention to the CUDA version of your installed driver. Your GPU's CUDA version must be equal to or greater than the CUDA version required by PyTorch. We strongly recommend using a GPU with the given code. You can check your version with this command:

<!-- Najprej se prepričajte, da ste namestili enega od najnovejših gonilnikov; v nasprotnem primeru imate lahko težave z razpoložljivostjo GPU-ja s knjižnico PyTorch. Bodite pozorni na nameščenega različico CUDA gonilnika. CUDA različica vašega GPU-ja mora biti enaka ali večja od različice CUDA, ki jo zahteva PyTorch. Priporočamo uporabo GPU-ja z dano kodo. Svojo različico lahko preverite s tem ukazom: -->

    nvidia-smi

For PyTorch follow the instructions here https://pytorch.org/get-started/locally/. If you are planning to build pythorch3d library for an unsupported version of Python, make sure you download the same version of Cuda Toolkit as your installed PyTorch library supports. The Cuda toolkit can be downloaded here https://developer.nvidia.com/cuda-toolkit-archive/.

<!-- Za PyTorch sledite navodilom tukaj https://pytorch.org/get-started/locally/. Če nameravate zgraditi knjižnico pythorch3d za nepodprto različico Pythona, se prepričajte, da ste prenesli isto različico Cuda Toolkita, kot jo podpira nameščena knjižnica PyTorch. Komplet orodij Cuda lahko prenesete tukaj https://developer.nvidia.com/cuda-toolkit-archive/. -->

Before we can install pytorch3d we need to install some additional libraries. Here we will install all necessary libraries for our code:

<!-- Preden lahko namestimo pytorch3d, moramo namestiti nekaj dodatnih knjižnic. Tukaj bomo namestili vse potrebne knjižnice za našo kodo: -->

    pip install torchinfo scipy seaborn scikit-learn pandas matplotlib k3d ipywidgets watermark ipykernel pyproject.toml wheel

For pytorch3d there are many ways to install it. We went with the one that is supported for Windows. For other ways of installation, you can check here https://github.com/facebookresearch/pytorch3d?tab=readme-ov-file#installation.

<!-- Za pytorch3d obstaja veliko načinov za namestitev. Šli smo s tistim, ki je podprt za Windows. Za druge načine namestitve lahko preverite tukaj https://github. com/facebookresearch/pytorch3d?tab=readme-ov-file#installation. -->

If you are installing it under Windows, you will also need Git, can be accessed here https://git-scm.com/download/win, and C++ Build Tools, which can be accessed here https://visualstudio.microsoft.com/visual-cpp-build-tools/. In C++ Build Tools, make sure to check and install "Desktop development with C++". You might need to reboot or close the current terminal in order for Git's path to be updated.

<!-- Če ga nameščate v operacijski sistem Windows, boste potrebovali tudi Git, do katerega lahko dostopate tukaj https://git-scm.com/download/win, in orodja za gradnjo C ++, do katerih lahko dostopate tukaj https://visualstudio.microsoft.com/visual-cpp-build-tools/. V orodjih C ++ Build preverite in namestite Desktop development with C++«. Morda boste morali znova zagnati ali zapreti trenutni terminal, da bo pot Git posodobljena. -->

After that we can install pytorch3d from git:

<!-- Po tem lahko namestimo pytorch3d iz git: -->

    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

Building pytorch3d with the upper command took around 8 minutes to complete on Ryzen 7.

<!-- Gradnja knjižnice pytorch3d je z zgornjim ukazom na sistemu z Ryzen 7 trajala približno 8 minut. -->

## Acknowledgments

We would like to express our gratitude to the authors of the neural networks that were used in our master's thesis and to the people who implemented them before us. It made our implementation easier when there was not a lot of information available from official sources.

- [PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies](https://doi.org/10.48550/arXiv.2206.04670)

- Unofficial implementations of PointNeXt:
    - [eat-slim](https://github.com/eat-slim/PointNeXt_pure_python)
    - [kentechx](https://github.com/kentechx/pointnext)

- [EfficientNetV2: Smaller Models and Faster Training](https://doi.org/10.48550/arXiv.2104.00298)

- Unofficial implementations of EfficientNetV2:
    - [PyTorch](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)

We would also like to express gratitude to the authors of publicly available datasets since finding one that is publicly available can be a challenge.

- Boni Vicari et al.: [Leaf and wood classification framework for terrestrial LiDAR point clouds: Simulated data validation dataset.](https://doi.org/10.5281/zenodo.1324158)

- Owen et al.: [Individual TLS tree clouds collected from both Alto Tajo and Cuellar in Spain.](https://doi.org/10.5281/zenodo.6962717)

- Seidel: [Single tree point clouds from terrestrial laser scanning.](https://doi.org/10.25625/FOHUJM)

## Dataset

Used dataset can be accessed from [here](https://e.pcloud.link/publink/show?code=XZPHjTZgj6bF6qjWJhfHDqAgQS2fJbycXLV) or [here](https://www.mediafire.com/file/4wgw4cb2n9g4nz1/archive.tar.gz/file). The download size is about 3.4 GiB, with a final size of up to 12 GiB.

## Our reference

Master's thesis and full reference can be accessed here.