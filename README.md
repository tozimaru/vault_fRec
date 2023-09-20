# vault_fRec

Face Recognition API for Vault. You are able to register faces, search for identities based on image URLs, and retrieve a list of registered users. **Deleting a registered user not implemented**.  Requires all uploaded images to be a jpeg.

## API Endpoints

### 1. Index (`/`)

- **Method**: `GET`
- **Response**: 
  - `200 OK` with a JSON message: `{'message': 'Vault Face API.'}`

### 2. Get Identity (`/get_identity`)

- **Method**: `POST`
- **Request Body**:
  - `urls`: List of image URLs to be processed.
  - `match_type`: Choose one of `default` or `strict`. This changes the **threshold** when matching faces.
  
- **Response**:
  - `200 OK` if identity found.
    - Returns a JSON object with `status`, `message`, and `details` about the processed images.
  - `400 Bad Request` if unable to download any images.
  
### 3. Get Registered Users (`/get_registered_users`)

- **Method**: `GET`
- **Response**:
  - `400 Bad Request`
    - Returns a JSON object with the number of registered users and a list of their IDs.

### 4. Register User (`/register_user`)

- **Method**: `POST`
- **Request Body**:
  - `urls`: List of image URLs to be processed.
  - `member_id`: ID of the member to be registered.
- **Response**:
  - `200 OK` if member registered successfully.
    - Returns a JSON object with `status`, `message`, and `details` about the processed images.
  - `400 Bad Request` if member already exists or unable to find faces in all images.

## Success and Failure:

1. **Success Scenarios**:
   - Successfully identified user's information.
   - Image processed but the individual in the image is unidentified.
   - Successfully added new user.

2. **Failure Scenarios**:
   - Cannot download any images from provided URLs.
   - Cannot find a face in the image.
   - Multiple faces found in the image.
   - Member ID already exists when trying to register a new user.

## What's in "details"?:
In "details", we have the following:
  - `invalid_urls`: List of urls that we couldn't download images from
  - `one_face_found`: List of urls we found exactly 1 face in. This is good.
  - `no_face_found`: List of urls we found NO faces in. This is bad.
  - `multiple_faces_found`: List of urls we found more than 1 face in. This is bad.