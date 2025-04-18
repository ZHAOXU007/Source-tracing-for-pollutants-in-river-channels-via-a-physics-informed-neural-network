# Source-tracing-for-pollutants-in-river-channels-via-a-physics-informed-neural-network
One-dimensional river traceability under cross-section data by PINN
Source tracing for pollutants in river channels via a physics-informed neural network
Xu Zhao1, Haifei Liu1*, Fei Leng1, Wei Yang1, Xinan Yin1
The river pollutant traceability problem represents a critical challenge in environmental monitoring and water resource management. In this work, we propose an approach based on a physics-informed neural network (PINN) for identifying key parameters of pollutant sources, including the release intensity and location, on the basis of cross-sectional observations. The accuracy of the proposed method was validated through experiments conducted on steady, unsteady, and noisy unsteady flows, with real-world river systems as test cases. The results demonstrate that the method can not only accurately identify source parameters beyond the gauging river reach but also effectively capture the spatial distribution of pollutants across the entire computational domain. This approach provides a novel solution for addressing the challenges of tracing pollutant sources in river channels.
net.py: network architecture file
mian.py: training file
visualisation.py: visualization file.
data3: observation data file
data3real: CFD data of the whole concentration field.
model: parameterized model
training_log.xlsx: training process file.
