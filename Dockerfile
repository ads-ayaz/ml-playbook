FROM jupyter/tensorflow-notebook
WORKDIR /home/jovyan
COPY ./ads_tf2_p36_spark3.txt .
RUN conda update conda
RUN pip install -U pip
#RUN while read requirement; do conda install --yes $requirement || pip install $requirement; done < ads_tf2_p36_spark3.txt
#RUN conda env create --file ads_tf2_p36_spark3.txt
CMD [ "start.sh", "jupyter", "lab" ]
