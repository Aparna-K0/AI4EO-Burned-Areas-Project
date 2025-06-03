# AI4EO-Burned-Areas-Project

This is a student-level project designed to analyse a wildfire-affected area using a trained CNN model and detect burned and unburned areas.

## Table of Contents

- [Video Guide](#video-guide)
- [Problem Description](#problem-description)
- [Project Goals](#project-goals)
- [Sentinel-2 and CNN Models](#sentinel-2-and-cnn-models)
- [Environmental Assessment](#environmental-assessment)
- [Getting Started](#getting-started)
  - [Set Up](#set-up)
  - [Data and Requirements](#data-and-requirements)
- [Acknowledgments](#acknowledgments)
- [References](#references)
- [Contact](#contact)
- [License](#license)
  
## Video Guide

I've made a video that runs through and explains this project, specifically the notebook: https://youtu.be/jWUL9ZuWu_E 
Some minor edits have been made since the video was recorded so please also check the notebook. 


## Problem Description 

As global temperatures rise, hot dry conditions are becoming more frequent, which in turn increases the frequency of wildfires, which pose a significant threat to ecosystems, infrastructure, and human life (World Health Organization, 2023). The ability to rapidly identify burned areas immediately after a wildfire event would improve the efficiency of disaster response and aid distribution. 

More ‘hands on’ and traditional methods for detecting fire damage, such as field surveys or manually interpreting satellite images require significant manpower and are incredibly time consuming. These are also often not practical for large areas or situations that require frequent monitoring due to the scale of the required response (Chuvieco et al., 2019). They also require significant amounts of pre-planning and on-ground support which diverts resources from other vital crisis areas. Automating this process using machine learning can help improve aid distribution and improve immediate responses to wildfires. This project attempts to do that using a trained CNN model. 

Satellites like Sentinel-2 provide increased, public access to earth observation data, creating opportunities to automate burned/damaged area detection using spectral and spatial features. Sentinel-2’s MSI (Multi Spectral Instrument) provides multispectral imagery across the visible, near-infrared (NIR), and shortwave infrared (SWIR) wavelength ranges, which are particularly useful for analysing vegetation changes and impacts of fire (European Space Agency eoPortal, 2024).


## Project Goals

-	Detect fire-affected/burned areas using Sentinel-2 L2A imagery before and after a wildfire event.

-	Calculate standard indices such as the Normalized Burn Ratio (NBR) to help identify burned regions.

- Train a Convolutional Neural Network model (CNN) using a combination of spectral bands and indices to identify burned areas.

-	Apply the trained model to a second area, affected by a wildfire, to evaluate the model’s performance and identify areas for improvement.

-	Estimate the total burned area in km² based on model predictions.
  
-	Ultimately, produce an automated method for detecting burned areas using only post-fire Sentinel-2 data.


## Sentinel-2 and CNN Models

![convolutional neural network (CNN)](https://github.com/user-attachments/assets/d03b3248-b470-466d-998e-128c3b172be8)
Sentinel-2 is part of ESA's Copernicus Constellation, aiming to provide high-resolution optical imagery for land monitoring. The mission focuses on vegetation, soil and water cover but has historically been used for a variety of monitoring purposes. The satellites operate at an altitude of just above 700km for optimal lighting.  

The Multi-Spectral Instrument (MSI) is a push-broom sensor that captures imagery in 13 spectral bands, including the visible, near-infrared (NIR), and short-wave infrared (SWIR) regions of the electromagnetic spectrum. This project selectively uses the Green (B03), Red(B04), NIR (B08) and SWIR (B11) bands. Bands have a variety of resolutions: 10m, 20m and 60 making them useful for a variety of research projects (European Space Agency eoPortal, 2024). 

![Copy of convolutional neural network (CNN)](https://github.com/user-attachments/assets/0ce4ed1f-85f7-42f2-bd2d-6be8f5b13c01)
Convolutional Neural Networks (CNNs) are a class of deep learning models designed to process data, such as satellite imagery. They're used for tasks that require object recognition, including image classification. This makes a CNN model very useful for a project like this. CNNs can automatically identify features reducing the need for manual identification. This is especially useful for large images such as those from Sentinel-2's MSI. This also directly tackles one of the problems mentioned earlier where manual satellite image analysis is a drain on time and resources (Keita, 2023). 


CNNs are made of four key components (Keita, 2023):
    - Convolutional layers -> Apply filters/kernals to the data to extract features like texture, shape and edges. 
    - Rectified Linear Unit (ReLU) -> Allow the model to learn compelex patterns
    - Pooling layers -> Decreases computational load by reducing spatial dimensions and preserving important features, e.g max pooling and average pooling. 
    - Fully connected layers (Dense layers) -> Take the filtered data and converts it to the final output, in this case feature classification


## Environmental Assessment

A growing concern with AI in general is the environmental impact, particularly when it comes to training deep learning models. Models like OpenAI’s GPT or large vision transformers can use Megawatt/hours worth of energy on single tasks (Vincent, 2024). 

Training for this projects’ CNN model was performed on a single A100 GPU using Google Colab with each full notebook run taking approximately 10–15 minutes, using only 20 epochs due to RAM restrictions (Google Cloud, 2025). The CNN model processes small 32×32 patches with 6 channels. As a result, energy costs are lower than those of large transformer models. The limited number of epochs and short training duration minimise computational demands. As a result, this is a more light and efficient application of AI, but it would be much more environmentally taxing if more areas were used for training or if the model was applied to a larger area than the 200 by 300 km training area used in this project. 

CO2 emissions for a project are based on a few factors such as hardware type, provider, length of usage and region of compute. The notebook ran for about six hours cumulatively while I worked on it so about 0.93 kg of CO2 (and equivalents) was emitted based on the notebook alone (“Machine Learning CO2 Impact Calculator,” n.d.). 

Water usage is another significant environmental cost. Google's data centres rely on evaporative cooling systems that consume 2-4 litres of water per kWh of energy used (Nicoletti et al., 2025). For this project's estimated energy use, 0.4-0.6 litres of water were used. This is negligible at a small scale but could be devastating when scaled up, potentially leading to water stress. 


## Getting Started 

1) Download ml_cnn_burned__area_usa_fires_2025 and save in Google Drive
2) Download the Sentinel-2 files referenced later. You can exchange the Dragoon Mountains file for those from another wildfire area.
3) Edit the notebook as necessary for your files, specifically the file paths at the top of the notebook or within the band reading function. If using other datasets for other areas modify the file/plot names as well.
4) Run the notebook cells in sequence.
   
#### Set Up 
Packages: 
Install using pip
```
os  # standard library, no need to install
glob  # standard library, no need to install
cv2  # opencv-python
numpy
matplotlib
seaborn
rasterio
scikit-image
scikit-learn
tensorflow
```

When using Google Colab mount Google Drive using
```
from google.colab import drive
drive.mount('/content/drive')
```
#### Data and Requirements

Data for this project came from the Copernicus Browswer: https://browser.dataspace.copernicus.eu 
Due to the size of the files they haven't been included in this repo but they are available from the browser using a free account.

Using the 'Search' tab within the browser access the following files:

Los Angeles pre-fire:

S2A_MSIL2A_20250102T183751_N0511_R027_T11SMT_20250102T221646.SAFE 

Los Angeles post-fire:

S2C_MSIL2A_20250201T183631_N0511_R027_T11SMT_20250201T210315.SAFE

Dragoon Mountains post-fire:

S2B_MSIL2A_20250508T175909_N0511_R041_T12SWA_20250508T215203.SAFE

I also reorganised each of these files so that the bands were no longer in folders based on resolution and instead all in the IMG_DATA folder. 

## Acknowledgments

This project was created for GEOL0069 at University College London, taught by Dr. Michel Tsamados, Weibin Chen and Connor Nelson.

A previous project from this module was also used as reference: 
McKee, F. (2024). GitHub - captainbluebear/GEOL0069-ML-Inland-Water-Body-Detection: Using k-means classification to analyse SENTINEL-2 satellite data for inland water body detection. Retrieved from https://github.com/captainbluebear/GEOL0069-ML-Inland-Water-Body-Detection/tree/main

## References
Anne-Laure Ligozat. (2024, November 13). Generative AI: energy consumption soars. Retrieved from https://www.polytechnique-insights.com/en/columns/energy/generative-ai-energy-consumption-soars/

Chuvieco, E., Mouillot, F., van der Werf, G. R., San Miguel, J., Tanase, M., Koutsias, N., et al. (2019). Historical background and current developments for mapping burned area from satellite Earth observation. Remote Sensing of Environment, 225, 45–64. https://doi.org/10.1016/j.rse.2019.02.013

European Space Agency eoPortal. (2024). Copernicus: Sentinel-2. Retrieved May 29, 2025, from https://www.eoportal.org/satellite-missions/copernicus-sentinel-2#overview

Freie Universität Berlin. (n.d.). Sentinel 2. Retrieved from https://blogs.fu-berlin.de/reseda/sentinel-2/

Govedarica, M., Jakovljević, G., Alvarez-Taboada, F., & Kokeza, Z. (2020, May 13). Near Real-Time Burned Area Mapping Using Sentinel-2 Data. Retrieved May 29, 2025, from https://www.researchgate.net/publication/341286133_Near_Real-Time_Burned_Area_Mapping_Using_Sentinel-2_Data

Keita, Z. (2023, November 14). An Introduction to Convolutional Neural Networks (CNNs). Retrieved from https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns

Machine Learning CO2 Impact Calculator. (n.d.). Retrieved from https://mlco2.github.io/impact/#compute

Martineau, K. (2020, August 7). Shrinking deep learning’s carbon footprint. Retrieved from https://news.mit.edu/2020/shrinking-deep-learning-carbon-footprint-0807

McKee, F. (2024). GitHub - captainbluebear/GEOL0069-ML-Inland-Water-Body-Detection: Using k-means classification to analyse SENTINEL-2 satellite data for inland water body detection. Retrieved from https://github.com/captainbluebear/GEOL0069-ML-Inland-Water-Body-Detection/tree/main

Naser, M. Z., & Kodur, V. (2025). Vulnerability of structures and infrastructure to wildfires: a perspective into assessment and mitigation strategies. Natural Hazards. https://doi.org/10.1007/s11069-025-07168-5

Nicoletti, L., Ma, M., & Bass, D. (2025, May 8). How AI Demand Is Draining Local Water Supplies. Retrieved from https://www.bloomberg.com/graphics/2025-ai-impacts-data-centers-water-data/Vincent, J. (2024, February 16). How much electricity does AI consume? Retrieved from https://www.theverge.com/24066646/ai-electricity-energy-watts-generative-consumptionWorld Health Organization. (2023). Wildfires. Retrieved from https://www.who.int/health-topics/wildfires

Yilmaz, E. O., & Kavzoglu, T. (2024). Burned Area Detection with Sentinel-2A Data: Using Deep Learning Techniques with eXplainable Artificial Intelligence. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, X-5-2024, 251–257. https://doi.org/10.5194/isprs-annals-x-5-2024-251-2024

Zhang, S., Bai, M., Wang, X., Peng, X., Chen, A., & Peng, P. (2023). Remote sensing technology for rapid extraction of burned areas and ecosystem environmental assessment. PeerJ, 11, e14557–e14557. https://doi.org/10.7717/peerj.14557

## Contact

Aparna Karthikanand - aparna.karthikanand.22@ucl.ac.uk

## License
Distributed under the MIT License. See LICENSE.txt for more information. 
