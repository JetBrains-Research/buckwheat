int main(int argc, char* argv[])
{
  float mini = bs::Zero<float>();   // Initialize mini with zero
  float maxi = bs::Valmax<float>(); /* Initialize maxi with maxval */
  if(argc >= 2) mini = std::atof(argv[1]);
  if(argc >= 3) maxi = std::atof(argv[2]);
  bs::exhaustive_test<bs::pack<float>> ( mini
                              , maxi
                              , bs::acosd
                              , raw_acosd()
                              );

  return 0;
}
