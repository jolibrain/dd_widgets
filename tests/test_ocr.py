from dd_widgets import OCR


def test_ocr():

    create_service_body = {
        "description": "synthtxt",
        "mllib": "caffe",
        "model": {
            "create_repository": True,
            "repository": "/data1/xolive/models/synthtxt",
            "templates": "../templates/caffe/",
        },
        "parameters": {
            "input": {
                "bw": False,
                "connector": "image",
                "ctc": True,
                "db": True,
                "height": 80,
                "unchanged_data": True,
                "width": 220,
            },
            "mllib": {
                "activation": "relu",
                "autoencoder": False,
                "db": True,
                "dropout": 0.0,
                "gpu": True,
                "gpuid": 0,
                "layers": [],
                "mirror": False,
                "nclasses": 37,
                "rotate": False,
                "scale": 1.0,
                "template": "crnn",
            },
            "output": {"store_config": True},
        },
        "type": "supervised",
    }

    train_service_body = {
        "async": True,
        "data": ["tests/file.txt", "tests/file.txt"],
        "parameters": {
            "input": {"db": True, "shuffle": True, "test_split": 0.0},
            "mllib": {
                "class_weights": [],
                "gpu": True,
                "gpuid": 0,
                "net": {"batch_size": 32, "test_batch_size": 16},
                "resume": False,
                "solver": {
                    "base_lr": 0.0001,
                    "iter_size": 1,
                    "iterations": 10000,
                    "snapshot": 5000,
                    "solver_type": "AMSGRAD",
                    "test_initialization": False,
                    "test_interval": 1000,
                },
                "timesteps": 32,
            },
            "output": {"measure": ["acc"], "target_repository": ""},
        },
        "service": "synthtxt",
    }

    o = OCR(
        "synthtxt",
        training_repo="tests/file.txt",
        testing_repo="tests/file.txt",
        host="127.0.0.1",
        port=12345,
        model_repo="/data1/xolive/models/synthtxt",
        img_width=220,
        img_height=80,
        nclasses=37,
        template="crnn",
        solver_type="AMSGRAD",
    )

    assert o._create_service_body() == create_service_body
    assert o._train_service_body() == train_service_body
