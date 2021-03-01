#include <thread>
#include "img_prod_cons.hpp"

int main() {
    ImageConsProd image_cons_prod;    
    std::thread t1(&ImageConsProd::ImageProducer, std::ref(image_cons_prod));
    std::thread t2(&ImageConsProd::ImageConsumer, std::ref(image_cons_prod));
    t1.join();
    t2.join();
}
