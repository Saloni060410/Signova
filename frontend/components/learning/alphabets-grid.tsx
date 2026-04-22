"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"
import { X, Video } from "lucide-react"

// Verified Wikimedia Commons ASL alphabet thumbnail URLs (public domain)
const aslImageUrls: Record<string, string> = {
  A: "https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Sign_language_A.svg/250px-Sign_language_A.svg.png",
  B: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Sign_language_B.svg/250px-Sign_language_B.svg.png",
  C: "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Sign_language_C.svg/250px-Sign_language_C.svg.png",
  D: "https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/Sign_language_D.svg/250px-Sign_language_D.svg.png",
  E: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Sign_language_E.svg/250px-Sign_language_E.svg.png",
  F: "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Sign_language_F.svg/250px-Sign_language_F.svg.png",
  G: "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Sign_language_G.svg/250px-Sign_language_G.svg.png",
  H: "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Sign_language_H.svg/250px-Sign_language_H.svg.png",
  I: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Sign_language_I.svg/250px-Sign_language_I.svg.png",
  J: "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Sign_language_J.svg/250px-Sign_language_J.svg.png",
  K: "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Sign_language_K.svg/250px-Sign_language_K.svg.png",
  L: "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Sign_language_L.svg/250px-Sign_language_L.svg.png",
  M: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Sign_language_M.svg/250px-Sign_language_M.svg.png",
  N: "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Sign_language_N.svg/250px-Sign_language_N.svg.png",
  O: "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Sign_language_O.svg/250px-Sign_language_O.svg.png",
  P: "https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Sign_language_P.svg/250px-Sign_language_P.svg.png",
  Q: "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Sign_language_Q.svg/250px-Sign_language_Q.svg.png",
  R: "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Sign_language_R.svg/250px-Sign_language_R.svg.png",
  S: "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Sign_language_S.svg/250px-Sign_language_S.svg.png",
  T: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Sign_language_T.svg/250px-Sign_language_T.svg.png",
  U: "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Sign_language_U.svg/250px-Sign_language_U.svg.png",
  V: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Sign_language_V.svg/250px-Sign_language_V.svg.png",
  W: "https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Sign_language_W.svg/250px-Sign_language_W.svg.png",
  X: "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Sign_language_X.svg/250px-Sign_language_X.svg.png",
  Y: "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Sign_language_Y.svg/250px-Sign_language_Y.svg.png",
  Z: "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Sign_language_Z.svg/250px-Sign_language_Z.svg.png",
}

// ASL letter data with descriptions
const alphabetData = [
  { letter: "A", desc: "Make a fist with your thumb resting on the side of your index finger.", dynamic: false },
  { letter: "B", desc: "Hold your four fingers straight up together and tuck your thumb into your palm.", dynamic: false },
  { letter: "C", desc: "Curve your hand into a C shape with fingers together.", dynamic: false },
  { letter: "D", desc: "Touch your thumb to your middle, ring, and pinky fingers while pointing your index finger up.", dynamic: false },
  { letter: "E", desc: "Curl all fingers down to touch the thumb, making a claw-like shape.", dynamic: false },
  { letter: "F", desc: "Touch your thumb and index finger together in a circle while keeping other fingers straight up.", dynamic: false },
  { letter: "G", desc: "Point your index finger and thumb sideways, parallel to each other.", dynamic: false },
  { letter: "H", desc: "Point your index and middle fingers sideways together, with the thumb tucked.", dynamic: false },
  { letter: "I", desc: "Make a fist and extend only your pinky finger straight up.", dynamic: false },
  { letter: "J", desc: "Start with the 'I' handshape (pinky up), then trace a J shape downward in the air.", dynamic: true },
  { letter: "K", desc: "Point your index and middle fingers up in a V, with thumb touching the middle finger.", dynamic: false },
  { letter: "L", desc: "Extend your index finger up and thumb out to form an L shape.", dynamic: false },
  { letter: "M", desc: "Place your thumb under your first three fingers (index, middle, ring) while making a fist.", dynamic: false },
  { letter: "N", desc: "Place your thumb under your first two fingers (index and middle) while making a fist.", dynamic: false },
  { letter: "O", desc: "Curve all fingers to touch the thumb, forming a circular O shape.", dynamic: false },
  { letter: "P", desc: "Similar to K, but point your hand downward — index and middle fingers down with thumb between.", dynamic: false },
  { letter: "Q", desc: "Similar to G, but point your index finger and thumb downward.", dynamic: false },
  { letter: "R", desc: "Cross your index and middle fingers while making a fist.", dynamic: false },
  { letter: "S", desc: "Make a fist with your thumb closed over your fingers.", dynamic: false },
  { letter: "T", desc: "Place your thumb between your index and middle fingers while making a fist.", dynamic: false },
  { letter: "U", desc: "Point your index and middle fingers straight up together.", dynamic: false },
  { letter: "V", desc: "Hold up your index and middle fingers in a V shape.", dynamic: false },
  { letter: "W", desc: "Hold up your index, middle, and ring fingers spread apart.", dynamic: false },
  { letter: "X", desc: "Make a fist and bend your index finger into a hook shape.", dynamic: false },
  { letter: "Y", desc: "Extend your thumb and pinky finger while keeping other fingers in a fist.", dynamic: false },
  { letter: "Z", desc: "Point your index finger and trace the letter Z in the air.", dynamic: true },
]

// Fallback MP4 URLs for dynamic letters until API loads
const fallbackDynamicVideoIds: Record<string, string> = {
  J: "https://media.spreadthesign.com/video/mp4/13/alphabet-letter-600-1.mp4", 
  Z: "https://media.spreadthesign.com/video/mp4/13/alphabet-letter-616-1.mp4",
}

export function AlphabetsGrid() {
  const [selectedLetter, setSelectedLetter] = useState<typeof alphabetData[0] | null>(null)
  const [dynamicVideoIds, setDynamicVideoIds] = useState<Record<string, string>>(fallbackDynamicVideoIds)

  useEffect(() => {
    fetch("http://localhost:8000/learning/videos?category=alphabet")
      .then(res => res.json())
      .then(data => {
        if (data && data.videos && data.videos.length > 0) {
          const videoMap: Record<string, string> = {}
          data.videos.forEach((v: any) => {
            videoMap[v.title] = v.video_id
          })
          setDynamicVideoIds(videoMap)
        }
      })
      .catch(err => console.error("Failed to fetch dynamic learning videos:", err))
  }, [])

  return (
    <div>
      <div className="text-center mb-8">
        <h2 className="text-xl font-semibold text-foreground">
          Sign Language Alphabets
        </h2>
        <p className="mt-2 text-muted-foreground">
          Learn the ASL alphabet from A to Z. Letters with a{" "}
          <span className="inline-flex items-center gap-1 text-primary font-medium">
            <Video className="h-3.5 w-3.5" /> video icon
          </span>{" "}
          involve motion.
        </p>
      </div>

      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 xl:grid-cols-9 gap-4">
        {alphabetData.map((item, index) => (
          <motion.div
            key={item.letter}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: index * 0.02 }}
          >
            <Card
              className="group cursor-pointer bg-card/50 hover:bg-card border-border/50 hover:border-primary/50 hover:shadow-md transition-all duration-200"
              onClick={() => setSelectedLetter(item)}
            >
              <CardContent className="p-4 text-center">
                {/* ASL Sign Image */}
                <div className="relative aspect-square rounded-lg bg-gradient-to-br from-primary/10 to-accent/10 flex items-center justify-center mb-3 group-hover:from-primary/20 group-hover:to-accent/20 transition-colors overflow-hidden">
                  <img
                    src={aslImageUrls[item.letter]}
                    alt={`ASL sign for letter ${item.letter}`}
                    className="h-full w-full object-contain p-2 transition-transform duration-200 group-hover:scale-110 dark:brightness-0 dark:invert"
                    loading="lazy"
                  />
                  {/* Dynamic badge for J and Z */}
                  {item.dynamic && (
                    <div className="absolute top-1 right-1 bg-primary/90 text-primary-foreground rounded-full p-1">
                      <Video className="h-3 w-3" />
                    </div>
                  )}
                </div>

                {/* Letter */}
                <span className="text-lg font-bold text-foreground group-hover:text-primary transition-colors">
                  {item.letter}
                </span>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Detail Modal */}
      <AnimatePresence>
        {selectedLetter && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* Backdrop */}
            <motion.div
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
              onClick={() => setSelectedLetter(null)}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            />

            {/* Modal Content */}
            <motion.div
              className="relative bg-card border border-border rounded-2xl shadow-2xl max-w-lg w-full overflow-hidden z-10"
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
            >
              {/* Close Button */}
              <button
                onClick={() => setSelectedLetter(null)}
                className="absolute top-4 right-4 z-20 p-2 rounded-full bg-background/80 hover:bg-background border border-border/50 text-muted-foreground hover:text-foreground transition-colors"
              >
                <X className="h-4 w-4" />
              </button>

              <div className="p-6">
                {/* Letter Header */}
                <div className="text-center mb-4">
                  <span className="text-5xl font-black text-primary">
                    {selectedLetter.letter}
                  </span>
                  {selectedLetter.dynamic && (
                    <span className="ml-3 inline-flex items-center gap-1 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium">
                      <Video className="h-3.5 w-3.5" />
                      Dynamic Sign
                    </span>
                  )}
                </div>

                {/* Content: Image + optional video for dynamic */}
                <div className="space-y-4">
                  {/* Always show the sign image */}
                  <div className="flex justify-center">
                    <div className="w-56 h-56 rounded-2xl bg-gradient-to-br from-primary/5 to-accent/5 flex items-center justify-center border border-border/30">
                      <img
                        src={aslImageUrls[selectedLetter.letter]}
                        alt={`ASL sign for letter ${selectedLetter.letter}`}
                        className="h-48 w-48 object-contain dark:brightness-0 dark:invert"
                      />
                    </div>
                  </div>

                  {/* Video for dynamic letters (J and Z) */}
                  {selectedLetter.dynamic && (
                    <div className="relative aspect-video rounded-xl overflow-hidden bg-gradient-to-br from-muted to-muted/50 border border-border/30">
                      <video
                        src={dynamicVideoIds[selectedLetter.letter]}
                        title={`ASL sign for letter ${selectedLetter.letter} - Video demonstration`}
                        className="absolute inset-0 w-full h-full object-cover bg-black"
                        autoPlay
                        loop
                        muted
                        playsInline
                        controls
                      />
                    </div>
                  )}

                  <div className="bg-muted/50 rounded-xl p-4">
                    <h3 className="text-sm font-semibold text-foreground mb-1">How to sign:</h3>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {selectedLetter.desc}
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
