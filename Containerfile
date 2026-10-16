# One image for the operator, gateway, verifier, detector and init helpers.
FROM registry.access.redhat.com/ubi9/python-311:latest

WORKDIR /opt/app-root/src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dualkey ./dualkey
# make the code readable whatever permissions the files had on the build machine
USER 0
RUN chmod -R a+rX /opt/app-root/src/dualkey

# OpenShift runs containers with a random UID in group 0
USER 1001
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/app-root/src

CMD ["python", "-m", "dualkey.detector"]
