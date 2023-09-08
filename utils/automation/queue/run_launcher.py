import argparse
from utils.automation.queue import QueueManager, JobLauncher

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--makedirs', dest='makedirs',
        help='enable ConfigLauncher to create missing directories',
        action='store_true'
    )
    parser.set_defaults(makedirs=False)
    parser.add_argument('--basepath', type=str, help='base path to use for ConfigLauncher', required=True)
    args = vars(parser.parse_args())

    base_path = args['basepath']
    queue = QueueManager(base_path, make_directories=args['makedirs'])
    launcher = JobLauncher(queue)
    launcher.run()
