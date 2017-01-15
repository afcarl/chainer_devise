#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys,os
from urllib import urlopen
from urllib import urlretrieve
from urllib2 import URLError,HTTPError 
import urllib2
import commands
import subprocess
import argparse
import random
from PIL import Image
import os.path
import time
import ssl

def cmd(cmd):
    return commands.getoutput(cmd)

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',        type=str,   default='images')
parser.add_argument('--num_of_classes',  type=int,   default=1000)
parser.add_argument('--num_of_pics',   type=int,   default=10)

args = parser.parse_args()

dict={}
c = 0
for line in open('words.txt', 'r'):
    line=line.split("\t")
    index = line[0]
    labels = [token.strip() for token in line[1].split(",")]
    first_label = labels[0].replace(" ", "_")
    dict[index]=first_label
    c += 1
assert (c == len(dict)), ""
print(dict["n07750586"])
#with open("./dict.txt", "w") as fin:
#    for (k, v) in dict.items():
#        fin.write("{k} {v}\n".format(k=k, v=v))

sys.exit()
ids = open('imagenet.synset.obtain_synset_list', 'r').read()
ids = ids.split()
random.shuffle(ids)

start = time.time()
cmd("mkdir %s"%args.data_dir)
for i in range(args.num_of_classes):
    id = ids[i].rstrip()
    category = dict[id]
    cnt = 0
    if len(category)>0:
        cmd("mkdir %s/%s"%(args.data_dir,category))
        print(category)
        sys.stdout.flush()
        try:
            urls=urlopen("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="+id).read()
            urls=urls.split()
            random.shuffle(urls)

            j=0
            while cnt<args.num_of_pics if args.num_of_pics<len(urls) else len(urls):
                url = urls[j]
                j+=1
                if j>=len(urls):
                    break
                print("url", url)
                sys.stdout.flush()

                filename = os.path.split(url)[1]
                try:
                    output = "%s/%s/%d_%s"%(args.data_dir,category,cnt,filename)
                    sys.stdout.flush()

                    #urlretrieve(url,output)
                    request = urllib2.urlopen(url, timeout=500)
                    with open(output, "wb") as f:
                        try:
                            f.write(request.read())
                        except IOError, e:
                            print e.reason
                            sys.stdout.flush()

                    try:
                        img = Image.open(output)
                        size = os.path.getsize(output)

                        if size==2051: #flickr Error
                            cmd("rm %s"%output)
                            cnt-=1                          
                    except IOError:
                        cmd("rm %s"%output)
                        cnt-=1
                except HTTPError, e:
                    cnt-=1
                    print "HTTPError", e.reason
                    sys.stdout.flush()
                except URLError, e:
                    cnt-=1
                    print "URLError", e.reason
                    sys.stdout.flush()
                except IOError, e:
                    cnt-=1
                    print "IOError", e
                    sys.stdout.flush()
                except ssl.CertificateError, e:
                    cnt-=1
                    print "ssl.CertificateError", e
                    sys.stdout.flush()
                except:
                    cnt-=1
                    print "unknown error"
                    sys.stdout.flush()
                cnt+=1
        except HTTPError, e:
            print e.reason
            sys.stdout.flush()
        except URLError, e:
            print e.reason
            sys.stdout.flush()
        except IOError, e:
            print e
            sys.stdout.flush()
end = time.time()
diff_sec = end - start
diff_min = diff_sec/60
diff_hour = diff_min/60
print("{}[sec]".format(diff_sec))
print("{}[min]".format(diff_min))
print("{}[hour]".format(diff_hour))
sys.stdout.flush()

