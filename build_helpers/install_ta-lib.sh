if [ -z "$1" ]; then
  INSTALL_LOC=/usr/local
else
  INSTALL_LOC=${1}
fi
echo "Installing to ${INSTALL_LOC}"
if [ ! -f "${INSTALL_LOC}/lib/libta_lib.a" ]; then
  wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz 2>&1 > /dev/null
  tar xvzf ta-lib-0.4.0-src.tar.gz 2>&1 > /dev/null
  cd ta-lib # Can't use !cd in co-lab
  ./configure --prefix=/usr 2>&1 > /dev/null
  make 2>&1 > /dev/null
  make install 2>&1 > /dev/null
  cd ../
else
  echo "TA-lib already installed, skipping installation"
fi
#  && sed -i.bak "s|0.00000001|0.000000000000000001 |g" src/ta_func/ta_utility.h \
