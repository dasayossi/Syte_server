import unittest
import os
from time import sleep
from multiprocessing import Pool
from src.syte_client import syte_client

def ask_for_prediction(client):
    return client.get_prediction()


class TestServer(unittest.TestCase):

    def test_health(self):

        client = syte_client()
        h = client.check_server_health()
        self.assertIsInstance(h, dict, 'Server is down')
        self.assertEqual(h['message'], 'Healthy', 'Server is down')

    def test_predict(self):
        client = syte_client()
        client.set_image('../data/image.jpg')
        p = client.get_prediction()
        self.assertIsInstance(p, str, 'Failed to get prediction')

    def test_parallel_prediction_request(self):
        clients_list = []
        for i in range(10):
            client = syte_client(client_name=str(i))
            client.set_image('../data/image.jpg')
            clients_list.append(client)

        p = Pool()
        results = p.map(ask_for_prediction, clients_list)
        p.close()
        p.join()
        for res in results:
            self.assertIsInstance(res, str, 'Unable to handle requests in parallel')


if __name__ == '__main__':

    # Turn on the server first
    os.system("start cmd /c python ../src/syte_server.py")
    # Wait enough time to ensure the server is up
    sleep(15)

    unittest.main()
