# production.wsgi
import sys
sys.path.insert(0,"/var/www/html/imginference/")
from run import app as application
