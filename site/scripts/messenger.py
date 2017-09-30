import redis
import json
import datetime
SERVER = redis.StrictRedis(host='localhost', port=6379, db=0)
def send_update(dataset_id, status):
    status.update({"time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    if 'error' in status['status']:
        SERVER.hset('update_errors', dataset_id, json.dumps(status))
    else:
        SERVER.hset('updates', dataset_id, json.dumps(status))