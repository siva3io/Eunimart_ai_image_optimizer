import logging
from flask import Blueprint, jsonify, request
from app.services.image_optimizer import ImageOptimizer
from app.services.image_optimizer.image_optimizer_new import ImageOptimizer_new
from app.core import limiter
from flask_limiter.util import get_remote_address
import flask_limiter


image_optimizer = Blueprint('image_optimizer', __name__)

logger = logging.getLogger(__name__)

@image_optimizer.route('/optimize', methods=['POST'])
def image_optimization():
    request_data = request.get_json()
    data = ImageOptimizer.optimize(request_data)
    if not data:
        data = {}
    return jsonify(data)

@image_optimizer.route('/optimize_new', methods=['POST'])
@limiter.limit('5/day',key_func = flask_limiter.util.get_ipaddr)

def image_optimization_new():
    request_data = request.get_json()
    data = ImageOptimizer_new.optimize(request_data)
    if not data:
        data = {}
    return jsonify(data)