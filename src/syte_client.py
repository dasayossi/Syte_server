import io
import requests
from PIL import Image
from logging import getLogger


class syte_client:

    def __init__(self, image_path=None, addr='http://localhost:5000', client_name='0'):

        self.logger = getLogger(client_name)#, level='INFO')
        self.addr = addr
        self.headers = {'content-type': 'image/jpeg'}
        self.image = None
        self.contents = None
        if image_path is not None:
            self.set_image(image_path)

    def get_prediction(self):

        self.response = requests.post(self.addr + '/predict', data=self.contents, headers=self.headers)
        return self.response.json()

    def set_image(self, image_path):

        try:
            self.image = Image.open(image_path)
        except:
            self.logger.error('Failed reading image')
            raise

        if self.image is None:
            self.logger.error('Failed reading image')
            raise

        with io.BytesIO() as output:
            self.image.save(output, format="JPEG")
            self.contents = output.getvalue()

    def check_server_health(self):

        self.response = requests.get(self.addr + '/health')
        return self.response.json()


if __name__ == '__main__':

    c = syte_client()
    c.set_image('../data/image.jpg')
    p = c.get_prediction()
    print(p)

    c2 = syte_client()
    h = c.check_server_health()

