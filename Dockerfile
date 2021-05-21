FROM python:3.8.10
ENV TZ="Europe/Amsterdam"

ADD requirements.txt /
RUN pip install -r /requirements.txt
RUN python -c "import nltk;nltk.download('punkt')"

ADD src /src

EXPOSE 8888

ENTRYPOINT [ "sh", "-c", "cd src;python3 entrypoint.py" ]
