FROM python:3
# this is a dockerfile that imports `xsdaliclock-local.py` and `xsdaliclock-local.html` from the current directory
# and runs the python script after `pip install`ing the required packages in `requirements.txt`
COPY requirements.txt /
RUN pip install -r requirements.txt
COPY xsdaliclock-local.py /
COPY xsdaliclock-local.html /
CMD [ "python", "./xsdaliclock-local.py" ]