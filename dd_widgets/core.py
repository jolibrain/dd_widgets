# fmt: off

import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict

import requests


# fmt: on


JSONType = Dict[str, Any]


class TalkWithDD:
    """Mechanism for talking with dede server."""

    sname_url = "http://{host}:{port}/{path}/services/{sname}"

    def default_service_body(self) -> JSONType:
        return OrderedDict(
            [
                ("mllib", "caffe"),
                ("description", self.sname),
                ("type", "supervised"),
                (
                    "parameters",
                    {
                        "mllib": {"nclasses": 42},  # why not?
                        "input": {"connector": "csv"},
                    },
                ),
                (
                    "model",
                    {
                        "repository": self.model_repo.value,
                        "create_repository": True,
                    },
                ),
            ]
        )

    def create_service(self, *_) -> JSONType:
        host = self.host.value
        port = self.port.value

        body = self.default_service_body()
        logging.info(
            "Creating service '{sname}':\n {body}".format(
                sname=self.sname, body=json.dumps(body, indent=2)
            )
        )
        c = requests.put(
            self.sname_url.format(
                host=host, port=port, path=self.path.value, sname=self.sname
            ),
            json.dumps(body),
        )

        if c.json()["status"]["code"] != 201:
            logging.warning(
                "Reply from creating service '{sname}': {json}".format(
                    sname=self.sname, json=json.dumps(c.json(), indent=2)
                )
            )
            raise RuntimeError(
                "Error code {code}: {msg}".format(
                    code=c.json()["status"]["dd_code"],
                    msg=c.json()["status"]["dd_msg"],
                )
            )
        else:
            logging.info(
                "Reply from creating service '{sname}': {json}".format(
                    sname=self.sname, json=json.dumps(c.json(), indent=2)
                )
            )

        json_dict = c.json()
        if "head" in json_dict:
            self.status = json_dict["head"]
        return json_dict

    def _create(self, *_) -> JSONType:
        logging.info("Entering _create method")
        host = self.host.value
        port = self.port.value
        body = self._create_service_body()

        sname_dict = dict(
            host=host, port=port, path=self.path.value, sname=self.sname
        )

        logging.info("Sending request " + self.sname_url.format(**sname_dict))
        c = requests.get(self.sname_url.format(**sname_dict))
        logging.info(
            "Current state of service '{sname}': {json}".format(
                sname=self.sname, json=json.dumps(c.json(), indent=2)
            )
        )
        if c.json()["status"]["msg"] != "NotFound":
            logging.warning(
                (
                    "Since service '{sname}' was still there, "
                    "it has been fully cleared: {json}"
                ).format(sname=self.sname, json=json.dumps(c.json(), indent=2))
            )

        logging.info(
            "Creating service '{sname}':\n {body}".format(
                sname=self.sname, body=json.dumps(body, indent=2)
            )
        )
        c = requests.put(self.sname_url.format(**sname_dict), json.dumps(body))

        if c.json()["status"]["code"] != 201:
            logging.warning(
                "Reply from creating service '{sname}': {json}".format(
                    sname=self.sname, json=json.dumps(c.json(), indent=2)
                )
            )
            raise RuntimeError(
                "Error code {code}: {msg}".format(
                    code=c.json()["status"]["dd_code"],
                    msg=c.json()["status"]["dd_msg"],
                )
            )
        else:
            logging.info(
                "Reply from creating service '{sname}': {json}".format(
                    sname=self.sname, json=json.dumps(c.json(), indent=2)
                )
            )

    def run(self, *_) -> JSONType:
        self._create()
        return self.train(resume=False)

    def resume(self, *_) -> JSONType:
        self._create()
        return self.train(resume=True)

    def train(self, resume: bool=False, *_) -> JSONType:
        body = self._train_service_body()
        host = self.host.value
        port = self.port.value

        if resume is True:
            body['parameters']['mllib']['resume'] = True

        logging.info(
            "Start training phase: {body}".format(
                body=json.dumps(body, indent=2)
            )
        )
        c = requests.post(
            "http://{host}:{port}/{path}/train".format(
                host=host, port=port, path=self.path.value
            ),
            json.dumps(body),
        )
        logging.info(
            "Reply from training service '{sname}': {json}".format(
                sname=self.sname, json=json.dumps(c.json(), indent=2)
            )
        )

        json_dict = c.json()
        if "head" in json_dict:
            self.status = json_dict["head"]

        thread = threading.Thread(target=self.update_loop)
        thread.start()

        return json_dict

    def delete(self, *_) -> JSONType:

        request = self.sname_url.format(
            host=self.host.value,
            port=self.port.value,
            path=self.path.value,
            sname=self.sname,
        )
        c = requests.delete(request)
        logging.info(
            "Delete service {sname}: {json}".format(
                sname=self.sname, json=json.dumps(c.json(), indent=2)
            )
        )
        json_dict = c.json()
        if "head" in json_dict:
            self.status = json_dict["head"]
        return json_dict

    def lightclear(self, *_) -> JSONType:
        # check why, but it seems we want to be sure we don't call the
        # inherited version of self.create_service()
        try:
            TalkWithDD.create_service(self)
        except RuntimeError:
            pass
        request = (self.sname_url + "?clear=lib").format(
            host=self.host.value,
            port=self.port.value,
            path=self.path.value,
            sname=self.sname,
        )
        c = requests.delete(request)
        logging.info(
            "Clearing (light) service {sname}: {json}".format(
                sname=self.sname, json=json.dumps(c.json(), indent=2)
            )
        )

        json_dict = c.json()
        if "head" in json_dict:
            self.status = json_dict["head"]
        self.delete()

        return json_dict

    def hardclear(self, *_) -> JSONType:
        # check why, but it seems we want to be sure we don't call the
        # inherited version of self.create_service()
        try:
            TalkWithDD.create_service(self)
        except RuntimeError:
            pass
        request = (self.sname_url + "?clear=full").format(
            host=self.host.value,
            port=self.port.value,
            path=self.path.value,
            sname=self.sname,
        )
        c = requests.delete(request)
        logging.info(
            "Clearing (full) service {sname}: {json}".format(
                sname=self.sname, json=json.dumps(c.json(), indent=2)
            )
        )

        json_dict = c.json()
        if "head" in json_dict:
            self.status = json_dict["head"]

        self.delete()
        return json_dict

    def update_loop(self) -> None:
        self.on_start()
        while True:
            info = self.info()
            status = info["head"]["status"]

            if status == "finished":
                self.on_finished(info)
                break

            if status == "error":
                self.on_error(info)
                break

            self.on_update(info)
            time.sleep(1)

    def info(self, *_) -> JSONType:
        # TODO job number
        request = (
            "http://{host}:{port}/{path}/train?service={sname}&"
            "job=1&timeout=10".format(
                host=self.host.value,
                port=self.port.value,
                path=self.path.value,
                sname=self.sname,
            )
        )
        c = requests.get(request)
        logging.debug(
            "Getting info for service {sname}: {json}".format(
                sname=self.sname, json=json.dumps(c.json(), indent=2)
            )
        )

        json_dict = c.json()
        if "head" in json_dict:
            self.status = json_dict["head"]
        return json_dict
