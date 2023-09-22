import os
import glob
from PIL import Image
from io import BytesIO
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import numpy as np
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from logger import logger

import insightface
from insightface.app import FaceAnalysis

SECRET_KEY = os.environ.get('VAULT_SECRET_KEY')

def require_secret_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if request.headers.get('vault-secret-key') != SECRET_KEY:
            print('Secret key fail')
            logger.error("Secret key fail")
            return jsonify({'message': 'Forbidden: Invalid Secret Key'}), 403
        return f(*args, **kwargs)
    logger.info("Secret key success")
    return wrapper

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB, adjust as needed

def check_size(url):
    response = requests.head(url)
    file_size = int(response.headers.get('Content-Length', 0))
    if file_size > MAX_FILE_SIZE:
        logger.error(f"File size ({file_size} bytes) exceeds the limit of {MAX_FILE_SIZE} bytes at {url}")
        return False, f"File size ({file_size} bytes) exceeds the limit of {MAX_FILE_SIZE} bytes"
    logger.info("file size is good")
    return True


def load_embeddings(DBDIR='./storage/registered/'):
    global global_target_embeddings, global_target_ids
    try:
        global_target_embeddings = []
        global_target_ids = []
        all_members = [d for d in os.listdir(DBDIR) if os.path.isdir(os.path.join(DBDIR, d))]

        for member in all_members:
            npys = glob.glob(os.path.join(DBDIR, member, '*.npy'))
            for npy in npys:
                try:
                    embedding = np.load(npy)
                    global_target_embeddings.append(embedding)
                    global_target_ids.append(member)
                except Exception as inner_e:
                    logger.exception(f"failed to load embedding from {npy}: {inner_e}")

        global_target_embeddings = global_target_embeddings 
        global_target_ids = global_target_ids 
        print('Loaded embeddings.')
        logger.info('Loaded embeddings successfully')
        return
    except Exception as e:
        logger.exception(f"load embeddings failed {e}")

def add_member_locally(urls, images, member_id, face_app, SAVEDIR='./storage/registered/'):
    try:
        outdir = os.path.join(SAVEDIR, member_id)
        one_face_found = []
        no_faces = []
        multiple_faces = []
        for i, image in enumerate(images):
            try:
                outputs = face_app.get(image)
                url = urls[i]

                if len(outputs) == 1:
                    os.makedirs(outdir, exist_ok=True)            
                    #image_outpath = os.path.join(outdir, str(i) + '.jpg')
                    npy_outpath = os.path.join(outdir, str(i) + '.npy')
                    np.save(npy_outpath, outputs[0]['embedding'])
                    #cv2.imwrite(image_outpath, image)
                    one_face_found.append(url)

                    global_target_embeddings.append(outputs[0]['embedding'])
                    global_target_ids.append(member_id)

                #adding these just so the error states are clearer
                elif len(outputs) > 1:
                    multiple_faces.append(url)
                else:
                    no_faces.append(url)
            except Exception as inner_e:
                logger.exception(f"Exception at image index {i} for URL {urls[i]}: {inner_e}")

        logger.info(f"Member registration success: {member_id}")
        return one_face_found, no_faces, multiple_faces
    except Exception as e:
        logger.exception(f"Member registration failed {member_id}: {e}")

def search_cosine(urls, images, face_app, threshold=0.55, DBDIR='./storage/registered/'):
    try:
        one_face = []
        no_faces = []
        multiple_faces = []
        print(f'looking up from {len(global_target_embeddings)} embeddings.')
        logger.info(f"'looking up from {len(global_target_embeddings)} embeddings.")
        for i, image in enumerate(images):
            try:
                outputs = face_app.get(image)
                url = urls[i]
                if len(outputs) == 1:
                    one_face.append(url)
                    for k, target_embedding in enumerate(global_target_embeddings):
                        distance = cosine_similarity(outputs[0]['embedding'], target_embedding)
                        if distance > threshold:
                            logger.info(f"Found identity: {global_target_ids[k]} with distance {distance}")
                            print('Found identity: ', global_target_ids[k], distance)
                            return {'identity': global_target_ids[k], 
                                    'one_face_found': one_face,
                                    'no_face_found': no_faces,
                                    'multiple_faces_found': multiple_faces}

                if len(outputs) > 1:
                    multiple_faces.append(url)
                
                if len(outputs) == 0:
                    no_faces.append(url)
            except Exception as inner_e:
                logger.exception(f"Exception at image index {i} for URL {urls[i]}: {inner_e}")
                
        logger.info(f"Search complete with threshold {threshold}")
        return {'identity': None, 
                'one_face_found': one_face,
                'no_face_found': no_faces,
                'multiple_faces_found': multiple_faces}
        
    except Exception as e:
        logger.exception(f"Search failed: {e}")

def download_images(urls):
    images = []
    valid_urls = []
    invalid_urls = []
    for url in urls:
        try:
            if not check_size(url):
                logger.error(f"{url} exceeds the file size limit")
                continue
            response = requests.get(url, timeout=10)
            if 'image' not in response.headers.get('content-type', ''):
                print(f"ERROR: URL {url} does not appear to be an image.")
                logger.error(f"ERROR: URL {url} does not appear to be an image.")
                continue
            image = Image.open(BytesIO(response.content))
            image_np = np.array(image)
            image_np = image_np[..., ::-1]
            images.append(image_np)
            valid_urls.append(url)
        except requests.exceptions.RequestException as error:
            invalid_urls.append(url)
            print(f"ERROR: Cannot download image from {url}. Reason: {error}")
            logger.error(f"ERROR: Cannot download image from {url}. Reason: {error}")
            continue
        
    logger.info("Images downloaded successfully")   
    return images, valid_urls, invalid_urls

def cosine_similarity(a, b):
    dot_product = np.dot(a.T, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

print('Loading embeddings...')
load_embeddings()

print('Loading face recognition engine.')
logger.info("Loading face recognition engine.")
face_app = FaceAnalysis(providers=['CPUExecutionProvider'],  allowed_modules=['recognition', 'detection'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

app = Flask(__name__)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per hour"]
)

CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/')
@require_secret_key
def index():
    try:
        logger.info("Index accessed")
        return jsonify({'message': 'Vault Face API.'}), 200
    except Exception as e:
        logger.exception(f"failed at idnex: {e}")

@app.route('/get_identity', methods=['POST'])
@require_secret_key
def get_identity():
    try:
        data = request.json
        urls = data['urls']
        match_type = data['match_type']
        
        threshold = 0.70 if match_type == 'strict' else 0.55
        #try to download images
        images, valid_urls, invalid_urls = download_images(urls)

        if not images:
            logger.error("Unable to download any images")
            response = {
                "status": "failed",
                "message": "Unable to download any images."
            }
            return jsonify(response), 400
        
        results = search_cosine(valid_urls, images, face_app, threshold=threshold)

        response = {
            'status': 'success',
            'message': "Found identities",
            'identity': results['identity'],
            'details': {
                        "invalid_urls": invalid_urls,
                        "one_face_found": results['one_face_found'],
                        "no_face_found": results['no_face_found'],
                        "multiple_faces_found": results['multiple_faces_found']}
                    }
        logger.info("Successfully identified user")
        return jsonify(response), 200
    except Exception as e:
        logger.exception(f"Failed to identify user: {e}")
        

@app.route('/get_registered_users', methods=['GET'])
@require_secret_key
def get_registered_users():
    try:
        data = {'num_users': len(set(global_target_ids)), 'registered_users': list(set(global_target_ids))}
        response = {
            "status": "success",
            "message": data
        }
        logger.info("Successfully fetched the list of registered users")
        return jsonify(response), 200
    except Exception as e:
        logger.exception(f"Failed to get list of registered users: {e}")


@app.route('/register_user', methods=['POST'])
@require_secret_key
def register_user():
    try:
        data = request.json
        urls = data['urls']
        member_id = data['member_id']

        if member_id in global_target_ids:
            response = {
                "status": "failed",
                "message": "Member already exists."
            }
            return jsonify(response), 400

        #try to download images
        images, valid_urls, invalid_urls = download_images(urls)

        if not images:
            logger.error("Unable to download any images")
            response = {
                "status": "failed",
                "message": "Unable to download any images."
            }
            return jsonify(response), 400

        one_face, no_faces, multiple_faces = add_member_locally(valid_urls, images, member_id, face_app)

        if len(one_face) == 0:
            logger.error("Unable to find faces in all images.")
            response = {
                "status": "failed",
                "message": "Unable to find faces in all images.",
                "details":  {
                            "invalid_urls": invalid_urls,
                            "one_face_found": one_face,
                            "no_face_found": no_faces,
                            "multiple_faces_found": multiple_faces
                            }
            }
            return jsonify(response), 400

        response = {
            "status": "success",
            "message": f"Successfully registered member: {member_id}",
            "details": {
                        "invalid_urls": invalid_urls,
                        "one_face_found": one_face,
                        "no_face_found": no_faces,
                        "multiple_faces_found": multiple_faces
                        }
        }
        logger.info(f"Successfully registered user {data['member_id']}")
        return jsonify(response), 200
    except Exception as e:
        logger.exception(f"User registration failed: {e}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)