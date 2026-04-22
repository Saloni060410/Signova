"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"
import { X } from "lucide-react"

// Verified Wikimedia Commons ASL/LSQ number sign image URLs
const numberImageUrls: Record<number, string> = {
  1: "https://upload.wikimedia.org/wikipedia/commons/1/1a/LSQ_1.jpg",
  2: "https://upload.wikimedia.org/wikipedia/commons/c/cc/LSQ_2.jpg",
  3: "https://upload.wikimedia.org/wikipedia/commons/6/68/LSQ_3.jpg",
  4: "https://upload.wikimedia.org/wikipedia/commons/e/ef/LSQ_4.jpg",
  5: "https://upload.wikimedia.org/wikipedia/commons/f/f1/LSQ_5.jpg",
  6: "https://upload.wikimedia.org/wikipedia/commons/0/03/LSQ_6.jpg",
  7: "https://upload.wikimedia.org/wikipedia/commons/7/7a/LSQ_7.jpg",
  8: "https://upload.wikimedia.org/wikipedia/commons/c/c8/LSQ_8.jpg",
  9: "https://upload.wikimedia.org/wikipedia/commons/d/de/LSQ_9.jpg",
  10: "https://upload.wikimedia.org/wikipedia/commons/5/5b/LSQ_10.jpg",
}

const numberData = [
  { number: 1, desc: "Hold up your index finger with the rest of your fingers in a fist." },
  { number: 2, desc: "Hold up your index and middle fingers in a V shape, palm facing outward." },
  { number: 3, desc: "Hold up your thumb, index, and middle fingers." },
  { number: 4, desc: "Hold up four fingers (index through pinky) with your thumb tucked in." },
  { number: 5, desc: "Spread all five fingers wide open." },
  { number: 6, desc: "Touch your pinky finger to your thumb while keeping the other three fingers up." },
  { number: 7, desc: "Touch your ring finger to your thumb while keeping the other three fingers up." },
  { number: 8, desc: "Touch your middle finger to your thumb while keeping the other three fingers up." },
  { number: 9, desc: "Touch your index finger to your thumb while keeping the other three fingers up." },
  { number: 10, desc: "Make a thumbs-up with your fist and shake your hand slightly." },
]

export function NumbersGrid() {
  const [selectedNumber, setSelectedNumber] = useState<typeof numberData[0] | null>(null)

  return (
    <div>
      <div className="text-center mb-8">
        <h2 className="text-xl font-semibold text-foreground">
          Sign Language Numbers
        </h2>
        <p className="mt-2 text-muted-foreground">
          Learn to sign numbers from 1 to 10
        </p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4 max-w-3xl mx-auto">
        {numberData.map((item, index) => (
          <motion.div
            key={item.number}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: index * 0.05 }}
          >
            <Card
              className="group cursor-pointer bg-card/50 hover:bg-card border-border/50 hover:border-primary/50 hover:shadow-md transition-all duration-200"
              onClick={() => setSelectedNumber(item)}
            >
              <CardContent className="p-6 text-center">
                {/* ASL Number Sign Image */}
                <div className="aspect-square rounded-xl bg-gradient-to-br from-primary/10 to-accent/10 flex items-center justify-center mb-4 group-hover:from-primary/20 group-hover:to-accent/20 transition-colors overflow-hidden">
                  <img
                    src={numberImageUrls[item.number]}
                    alt={`ASL sign for number ${item.number}`}
                    className="h-full w-full object-cover rounded-lg transition-transform duration-200 group-hover:scale-110"
                    loading="lazy"
                  />
                </div>

                {/* Number */}
                <span className="text-2xl font-bold text-foreground group-hover:text-primary transition-colors">
                  {item.number}
                </span>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Detail Modal */}
      <AnimatePresence>
        {selectedNumber && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* Backdrop */}
            <motion.div
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
              onClick={() => setSelectedNumber(null)}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            />

            {/* Modal Content */}
            <motion.div
              className="relative bg-card border border-border rounded-2xl shadow-2xl max-w-md w-full overflow-hidden z-10"
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
            >
              {/* Close Button */}
              <button
                onClick={() => setSelectedNumber(null)}
                className="absolute top-4 right-4 z-20 p-2 rounded-full bg-background/80 hover:bg-background border border-border/50 text-muted-foreground hover:text-foreground transition-colors"
              >
                <X className="h-4 w-4" />
              </button>

              <div className="p-6">
                {/* Number Header */}
                <div className="text-center mb-4">
                  <span className="text-5xl font-black text-primary">
                    {selectedNumber.number}
                  </span>
                </div>

                {/* Large Image */}
                <div className="flex justify-center mb-4">
                  <div className="w-56 h-56 rounded-2xl bg-gradient-to-br from-primary/5 to-accent/5 flex items-center justify-center border border-border/30 overflow-hidden">
                    <img
                      src={numberImageUrls[selectedNumber.number]}
                      alt={`ASL sign for number ${selectedNumber.number}`}
                      className="h-full w-full object-cover rounded-xl"
                    />
                  </div>
                </div>

                {/* Description */}
                <div className="bg-muted/50 rounded-xl p-4">
                  <h3 className="text-sm font-semibold text-foreground mb-1">How to sign:</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {selectedNumber.desc}
                  </p>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
