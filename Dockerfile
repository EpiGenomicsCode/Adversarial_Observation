FROM  pytorch/pytorch

# install all default anaconda libraries
RUN conda install anaconda

# Updates as needed
RUN apt-get update
RUN apt-get install libxrender1 -y
RUN apt-get install -y libsm6 libxext6 -y
RUN apt-get install -y libxrender-dev -y
RUN conda install python-graphviz -y
RUN conda install -c conda-forge tensorflow -y
RUN pip install torchviz
RUN pip install tensorflow-model-optimization