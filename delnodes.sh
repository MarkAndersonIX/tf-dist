echo "unallocating nodes.."
for line in `cat joblist`; do qdel $line; done;
