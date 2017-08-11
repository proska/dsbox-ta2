"""
Command Line Interface for running the DSBox TA2 Planner
"""

# Setup path before loading any dsbox.* packages. Or, alternatively
# use ../dsbox-dev-setup.sh to setup PYTHONPATH.
from dsbox_dev_setup import path_setup
path_setup()

import sys
import os
import argparse
from dsbox.controller import Controller

__all__ = []
__version__ = 0.1
__date__ = '2017-06-27'
__updated__ = '2017-06-27'

DEBUG = 0
TESTRUN = 0
PROFILE = 0

class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''
    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg

def main(argv=None): # IGNORE:C0111
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s
USAGE
''' % program_shortdesc

    #try:
    # Setup argument parser
    parser = argparse.ArgumentParser(description=program_license, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--problem", dest="problem", help="Problem directory")
    parser.add_argument("-l", "--library", dest="library", help="Primitives library directory. [default: %(default)s]", default="library")
    parser.add_argument("-o", "--output", dest="output", help="Output directory. [default: %(default)s]", default="output")
    parser.add_argument("-v", "--verbose", dest="verbose", action="count", help="set verbosity level [default: %(default)s]")
    parser.add_argument('-V', '--version', action='version', version=program_version_message)

    # Process arguments
    args = parser.parse_args()

    problem = args.problem
    if not problem:
        sys.stderr.write(program_name + ": No problem directory specified\n")
        sys.stderr.write("  for help use --help\n")
        exit(1)

    library = args.library
    verbose = args.verbose
    output = args.output

    if verbose > 0:
        print("Verbose mode on")

    controller = Controller(problem, library, output)
    controller.start()

'''
    return 0
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception, e:
        print(e)
        if DEBUG or TESTRUN:
            raise(e)
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help")
        return 2
'''

if __name__ == "__main__":
    if DEBUG:
        sys.argv.append("-h")
        sys.argv.append("-v")
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'planner.run_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())
