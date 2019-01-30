var res = mldb.put("/v1/procedures/airline", {
    "type":"import.text",
    "params": {
        "dataFileUrl": "file://allyears.1987.2013.csv.lz4",
        //"dataFileUrl": "file://allyears.1987.2013.csv",
        "offset" : 0,
        "ignoreBadLines" : true,
        "outputDataset": {
            "id": "airline"
        },
        "runOnCreation": true,
        //"select": "* excluding (DepTime)"
        //"select": "ActualElapsedTime"
        "select": "*"
    }
})

mldb.log(res.json);

mldb.log(mldb.post('/v1/datasets/airline/routes/saves',
                   { dataFileUrl: 'file://airline.mldbds' }));

