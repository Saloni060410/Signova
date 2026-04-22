"use client"

import { HeroSection } from "@/components/home/hero-section"
import { HowItWorksSection } from "@/components/home/how-it-works"
import { DeveloperSection } from "@/components/home/developer-section"
import { FeedbackSection } from "@/components/home/feedback-section"

export default function HomePage() {
  return (
    <>
      <HeroSection />
      <HowItWorksSection />
      <DeveloperSection />
      <FeedbackSection />
    </>
  )
}
