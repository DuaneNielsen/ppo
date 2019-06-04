import logging

from services.coordinator import Coordinator
from services.gatherer import Gatherer
from services.tensorboard import TensorBoardListener
from services.trainer import Trainer
from argparse import ArgumentParser


if __name__ == '__main__':

    parser = ArgumentParser(description='Start server.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--trainer", help="start a training instance",
                       action="store_true")
    group.add_argument("-c", "--coordinator", help="start a coordinator instance",
                       action="store_true")
    group.add_argument("-g", "--gatherer", help="start a gathering instance",
                       action="store_true")
    group.add_argument("-m", "--monitor", help="start a monitoring instance",
                       action="store_true")
    group.add_argument("--start", help="start training",
                       action="store_true")
    group.add_argument("--stopall", help="stop training",
                       action="store_true")

    parser.add_argument("-rh", "--redis-host", help='hostname of redis server', dest='redis_host', default='localhost')
    parser.add_argument("-rp", "--redis-port", help='port of redis server', dest='redis_port', default=6379)
    parser.add_argument("-ra", "--redis-password", help='password of redis server', dest='redis_password', default=None)
    parser.add_argument("-rd", "--redis-db", help='db of redis server', dest='redis_db', default=0)

    parser.add_argument("-ph", "--postgres-host", help='hostname of postgres server', dest='postgres_host',
                        default='localhost')
    parser.add_argument("-pp", "--postgres-port", help='port of postgres server', dest='postgres_port', default=5432)
    parser.add_argument("-pd", "--postgres-db", help='hostname of postgres db', dest='postgres_db',
                        default='policy_db')
    parser.add_argument("-pu", "--postgres-user", help='hostname of postgres user', dest='postgres_user',
                        default='policy_user')
    parser.add_argument("-pa", "--postgres-password", help='password of postgres server', dest='postgres_password',
                        default='password')

    args = parser.parse_args()

    '''
    Logging is controlled from here
    '''
    logging.basicConfig(format='%(levelname)s-%(module)s-%(message)s', level=logging.INFO)
    logging.getLogger('peewee').setLevel(logging.INFO)
    logging.info(args)

    if args.trainer:
        trainer = Trainer(args.redis_host, args.redis_port, args.redis_db, args.redis_password)
        trainer.main()

    elif args.gatherer:
        gatherer = Gatherer(args.redis_host, args.redis_port, args.redis_db, args.redis_password)
        gatherer.main()

    elif args.monitor:
        tb = TensorBoardListener(args.redis_host, args.redis_port, args.redis_db, args.redis_password)
        tb.main()

    elif args.coordinator:
        co_ordinator = Coordinator(args.redis_host, args.redis_port, args.redis_db, args.redis_password,
                                   args.postgres_host, args.postgres_port, args.postgres_db, args.postgres_user,
                                   args.postgres_password)
        co_ordinator.main()
