from detection import detectImage
from PIL import Image

class TestFace(object):
    def test_face(self):
        image = Image.open("test_images/crosby-smile.jpg")
        response = detectImage(image)
        assert response['face']['emotion'] == "happy"
        assert 240 <= response['face']['height'] <= 280
        assert 240 <= response['face']['width'] <= 280 
        assert 65 <= response['face']['x'] <= 105 
        assert 110 <= response['face']['y'] <= 150

    def test_face_one_hand(self):
        image = Image.open("test_images/one-hand.jpg")
        response = detectImage(image)
        assert response['face']['emotion'] == "neutral"
        assert 810 <= response['face']['height'] <= 850
        assert 810 <= response['face']['width'] <= 850
        assert 1165 <= response['face']['x'] <= 1205 
        assert 320 <= response['face']['y'] <= 360

    def test_face_two_hands(self):
        image = Image.open("test_images/two-hands.jpg")
        response = detectImage(image)
        assert response['face']['emotion'] == "neutral"
        assert 860 <= response['face']['height'] <= 900
        assert 860 <= response['face']['width'] <= 900
        assert 930 <= response['face']['x'] <= 970
        assert 620 <= response['face']['y'] <= 660

    def test_no_face(self):
        image = Image.open("test_images/pikachu.jpg")
        response = detectImage(image)
        assert response['face']['emotion'] == None
        assert response['face']['height'] == None
        assert response['face']['width'] == None
        assert response['face']['x'] == None
        assert response['face']['y'] == None

class TestHands(object):
    def test_no_hands(self):
        image = Image.open("test_images/pikachu.jpg")
        response = detectImage(image)
        assert len(response['hands']) == 2
        assert response['hands'][0]['height'] == None
        assert response['hands'][0]['width'] == None 
        assert response['hands'][0]['x'] == None 
        assert response['hands'][0]['y'] == None
        assert response['hands'][1]['height'] == None
        assert response['hands'][1]['width'] == None 
        assert response['hands'][1]['x'] == None 
        assert response['hands'][1]['y'] == None

    def test_one_hand(self):
        image = Image.open("test_images/one-hand.jpg")
        response = detectImage(image)
        assert len(response['hands']) == 2
        assert 1140 <= response['hands'][0]['height'] <= 1180
        assert 715 <= response['hands'][0]['width'] <= 755
        assert 30 <= response['hands'][0]['x'] <= 70
        assert 770 <= response['hands'][0]['y'] <= 810
        assert response['hands'][1]['height'] == None
        assert response['hands'][1]['width'] == None 
        assert response['hands'][1]['x'] == None 
        assert response['hands'][1]['y'] == None

    def test_two_hands(self):
        image = Image.open("test_images/two-hands.jpg")
        response = detectImage(image)
        assert len(response['hands']) == 2
        assert 1200 <= response['hands'][0]['height'] <= 1240
        assert 850 <= response['hands'][0]['width'] <= 890
        assert 0 <= response['hands'][0]['x'] <= 20
        assert 940 <= response['hands'][0]['y'] <= 980
        assert 1230 <= response['hands'][1]['height'] <= 1270
        assert 830 <= response['hands'][1]['width'] <= 870
        assert 2070 <= response['hands'][1]['x'] <= 2110
        assert 840 <= response['hands'][1]['y'] <= 880

