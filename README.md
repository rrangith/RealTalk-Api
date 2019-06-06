# RealTalk-Api

## Docker

To build and run the docker images, run the follow commands. The API_KEY environment is optional, if none is set, no api key will be used.

```
docker build -t realtalk-api .
docker run -it -e API_KEY=[Insert Value Here] -p 5000:5000 realtalk-api
```

Example: `docker run -it -e API_KEY=asd -p 5000:5000 realtalk-api`
