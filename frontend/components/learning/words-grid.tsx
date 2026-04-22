"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"
import { Play, X, Video } from "lucide-react"
import { Button } from "@/components/ui/button"

// Fallback static words until API fetches
const defaultWords = [
  {
    word: "Hello",
    description: "Bring your dominant hand to your forehead/temple area and move it outward, like a salute.",
    videoId: "https://media.spreadthesign.com/video/mp4/13/109919.mp4",
    startTime: 0,
  },
  {
    word: "Thank You",
    description: "With your dominant hand flat, touch your fingertips to your chin and move your hand outward toward the person.",
    videoId: "https://media.spreadthesign.com/video/mp4/13/58476.mp4",
    startTime: 0,
  },
  {
    word: "Sorry",
    description: "Form an 'A' handshape (fist with thumb alongside) and rub it in a circular motion on the center of your chest.",
    videoId: "https://media.spreadthesign.com/video/mp4/13/94029.mp4",
    startTime: 0,
  },
  {
    word: "Please",
    description: "With your dominant hand flat, place it on your chest and move it in a circular motion.",
    videoId: "https://media.spreadthesign.com/video/mp4/13/49200.mp4",
    startTime: 0,
  },
]

export function WordsGrid() {
  const [playingVideo, setPlayingVideo] = useState<string | null>(null)
  const [words, setWords] = useState<typeof defaultWords>(defaultWords)

  useEffect(() => {
    fetch("http://localhost:8000/learning/videos?category=word")
      .then(res => res.json())
      .then(data => {
        if (data && data.videos && data.videos.length > 0) {
          const fetchedWords = data.videos.map((v: any) => ({
            word: v.title,
            description: v.description || "",
            videoId: v.video_id,
            startTime: v.start_time || 0
          }))
          setWords(fetchedWords)
        }
      })
      .catch(err => console.error("Failed to fetch words videos:", err))
  }, [])

  return (
    <div>
      <div className="text-center mb-8">
        <h2 className="text-xl font-semibold text-foreground">
          Dynamic Words
        </h2>
        <p className="mt-2 text-muted-foreground">
          Learn common words and phrases with video demonstrations
        </p>
      </div>

      <div className="grid sm:grid-cols-2 gap-6 max-w-4xl mx-auto">
        {words.map((item, index) => (
          <motion.div
            key={item.word}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: index * 0.1 }}
          >
            <Card className="group h-full bg-card/50 hover:bg-card border-border/50 hover:border-primary/50 hover:shadow-lg transition-all duration-200">
              <CardContent className="p-0">
                {/* Video Area */}
                <div className="relative aspect-video bg-gradient-to-br from-muted to-muted/50 rounded-t-lg overflow-hidden">
                  {playingVideo === item.word ? (
                    <>
                      <video
                        src={item.videoId}
                        title={`ASL sign for ${item.word}`}
                        className="absolute inset-0 w-full h-full object-cover bg-black"
                        autoPlay
                        loop
                        muted
                        playsInline
                        controls
                      />
                      <button
                        onClick={() => setPlayingVideo(null)}
                        className="absolute top-2 right-2 z-10 p-1.5 rounded-full bg-black/60 text-white hover:bg-black/80 transition-colors"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </>
                  ) : (
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                      <div className="flex items-center gap-2 text-muted-foreground/60">
                        <Video className="h-8 w-8" />
                      </div>
                      <Button
                        size="lg"
                        variant="secondary"
                        className="rounded-full h-14 w-14 p-0 shadow-lg group-hover:bg-primary group-hover:text-primary-foreground transition-colors"
                        onClick={() => setPlayingVideo(item.word)}
                      >
                        <Play className="h-6 w-6 ml-1" />
                        <span className="sr-only">Play video for {item.word}</span>
                      </Button>
                      <span className="text-xs text-muted-foreground/50 font-medium">
                        Click to watch demo
                      </span>
                    </div>
                  )}
                </div>

                {/* Content */}
                <div className="p-5">
                  <h3 className="text-lg font-semibold text-foreground mb-2">
                    {item.word}
                  </h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {item.description}
                  </p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
