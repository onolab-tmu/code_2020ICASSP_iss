# We will use Ubuntu for our image
FROM ubuntu:latest

# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade
RUN apt-get install -y emacs neovim git gcc g++

# Adding wget and bzip2
RUN apt-get install -y wget bzip2

# Add sudo
RUN apt-get -y install sudo

# Add user ubuntu with no password, add to sudo group
RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu/
RUN chmod a+rwx /home/ubuntu/
#RUN echo `pwd`

# Anaconda installing
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm Miniconda3-latest-Linux-x86_64.sh

# Set path to conda
#ENV PATH /root/anaconda3/bin:$PATH
ENV PATH /home/ubuntu/miniconda3/bin:$PATH

# Updating Anaconda packages
RUN conda update conda
RUN conda update --all

# Get package
RUN git clone --recursive https://github.com/onolab-tmu/code_2020ICASSP_iss.git
ENV PYTHONPATH=/home/ubuntu/code_2020ICASSP_iss:$PYTHONPATH

# Install environment
RUN cd code_2020ICASSP_iss
RUN conda env create -f piva/environment.yml
RUN conda activate piva
RUN cd piva
RUN python setup.py build_ext --inplace
RUN cd ..

# Run Jupytewr notebook as Docker main process
CMD [ "sh", "--login", "-i" ]
