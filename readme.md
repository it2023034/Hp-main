#run the api without docker
python -m uvicorn main:app --reload



#run the docker
docker build -t hpapi .
docker run -p 8000:8000 --name hpapi hpapi

#check inside the docker (exit with exit)
docker exec -it hpapi /bin/sh
