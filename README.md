# RealTalk-Api

https://api-realtalk.herokuapp.com/ (Due to limited memory from the free tier, this version is not consistent)

This API resolves an issue from [RealTalk](https://github.com/rrangith/realTalk/). That project used a server that saved the state. As a result, multiple people would not be able to use the app at the same time. With this microservice, multiple people will be able to use the app at the same time. Other users are also able to use this API in their apps to determine the coordinates of a face and hands in an image, along with emotions.

## API Docs

### Authentication

For all requests, you must pass in an `x-api-key` header with the api key. If building this on your own, you can set the api key yourself. If no api key is specified, you do not need to be authorized to use the api.

### Response

The JSON response will return two objects: `face` and `hands`. The API currently only supports 1 face and up to 2 hands.

`face` will contain `emotion, height, width, x, y`. The `x` and `y` coordinates are the bottom left corner of the face, and the height and width can be used to determine the other coordinates of the face. If no face is found, all values in `face` will be null.

`hands` will contain an array of 2. Each element in the array is a hand and will contain `height, width, x, y`. The `x` and `y` coordinates are the bottom left corner of the hand, and the height and width can be used to determine the other coordinates of the hand. If no hands are found, the values will all be null.

Example response for a face with one hand:

```
{
    "face": {
        "emotion": "neutral",
        "height": 831,
        "width": 831,
        "x": 1184,
        "y": 337
    },
    "hands": [
        {
            "height": 1159,
            "width": 734,
            "x": 52,
            "y": 787
        },
        {
            "height": null,
            "width": null,
            "x": null,
            "y": null
        }
    ]
}
```

#### POST: /detectLink

In your body, pass in the link of an image as `url`

```
{
    "url": "https://example.com/image.png"
}
```

#### POST: /detectFile

Pass in the file as the value, with the key `file`

## Multi Threading

One of the bottlenecks in this project was that it took a long time to detect the face and hands. The detection of the face and hands could be done in parallel, so I used two threads to do this detection concurrently. This resulted in faster responses.

## Automatic Builds

This project uses Travis CI to run unit tests from `tests.py` and deploy to heroku on updates to the master branch.

## Docker

To build and run the docker images, run the follow commands. The API_KEY environment is optional, if none is set, no api key will be used.

```

docker build -t realtalk-api .
docker run -it -e API_KEY=[Insert Value Here] -p 5000:5000 realtalk-api

```

Example: `docker run -it -e API_KEY=asd -p 5000:5000 realtalk-api`
