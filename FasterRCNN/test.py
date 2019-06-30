import optparse
parser = optparse.OptionParser()
parser.add_option('-n', '--new', help='creates a new object')

(opts, args) = parser.parse_args()
