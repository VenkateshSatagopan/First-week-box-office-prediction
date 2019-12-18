# backend

### Project setup
Install dependencies to your environment using requirements.txt

```
pip install -r requirements.txt
```
or install them one by one until all errors are gone.

### Run API server
```
python server.py
```

or open ```server.py``` in your IDE to run it manually

### Test API server
Visit [http://localhost:80/api/some_input] to see a response.

### Include frontend in your server
To serve both an API and the frontend you have to first build the frontend:

Navigate to ```/frontend``` and run
```
npm run build
```
the ```/frontend/dist``` folder created there will then also be served.

### Test frontend server
Visit [http://localhost:80] or just [http://localhost] to see the served frontend SPA
